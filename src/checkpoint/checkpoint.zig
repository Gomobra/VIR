// Checkpoint (PRD 3.4). Safetensors writer + atomic protocol.

const std = @import("std");

pub const SafetensorsHeader = struct {};

// Write a single f32 tensor to safetensors format.
pub fn writeF32Tensor(allocator: std.mem.Allocator, path: []const u8, name: []const u8, shape: []const usize, data: []const f32) !void {
    var numel: usize = 1;
    for (shape) |s| numel *= s;
    std.debug.assert(data.len == numel);

    var json_buf = std.ArrayList(u8).init(allocator);
    defer json_buf.deinit();
    try json_buf.appendSlice("{\""); 
    try json_buf.appendSlice(name);
    try json_buf.appendSlice("\":{\"dtype\":\"F32\",\"shape\":[");
    for (shape, 0..) |s, i| {
        if (i > 0) try json_buf.appendSlice(",");
        var tmp: [32]u8 = undefined;
        const sl = try std.fmt.bufPrint(&tmp, "{d}", .{s});
        try json_buf.appendSlice(sl);
    }
    try json_buf.appendSlice("],\"data_offsets\":[0,");
    var tmp: [32]u8 = undefined;
    const sl = try std.fmt.bufPrint(&tmp, "{d}", .{numel * 4});
    try json_buf.appendSlice(sl);
    try json_buf.appendSlice("]}}");

    const header_len = json_buf.items.len;
    const padded = (header_len + 7) & ~@as(usize, 7);

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, padded, .little);
    try file.writeAll(&len_bytes);

    try file.writeAll(json_buf.items);
    var pad: usize = padded - header_len;
    while (pad > 0) : (pad -= 1) try file.writeAll(" ");

    for (data) |v| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, @bitCast(v), .little);
        try file.writeAll(&b);
    }
}

// --- Atomic checkpoint protocol (PRD 3.4) ---

pub fn sha256FileHex(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const st = try file.stat();
    const buf = try allocator.alloc(u8, st.size);
    defer allocator.free(buf);
    _ = try file.readAll(buf);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(buf);
    const digest = h.finalResult();
    const hex = try allocator.alloc(u8, 64);
    const hexchars = "0123456789abcdef";
    for (digest, 0..) |byte, i| {
        hex[i * 2] = hexchars[byte >> 4];
        hex[i * 2 + 1] = hexchars[byte & 0x0f];
    }
    return hex;
}

// Atomic save: write meta.json to tmp dir, fsync, write COMMIT with sha256, rename to final, update latest.
pub fn atomicSave(allocator: std.mem.Allocator, base_dir: []const u8, step: u64, meta_json: []const u8) !void {
    // tmp dir name
    var tmp_name_buf: [128]u8 = undefined;
    const tmp_name = try std.fmt.bufPrint(&tmp_name_buf, "ckpt-{d}.tmp", .{step});
    var final_name_buf: [128]u8 = undefined;
    const final_name = try std.fmt.bufPrint(&final_name_buf, "ckpt-{d}", .{step});

    const tmp_path = try std.fs.path.join(allocator, &.{ base_dir, tmp_name });
    defer allocator.free(tmp_path);
    const final_path = try std.fs.path.join(allocator, &.{ base_dir, final_name });
    defer allocator.free(final_path);

    // create tmp dir
    try std.fs.cwd().makePath(tmp_path);

    // write meta.json
    const meta_path = try std.fs.path.join(allocator, &.{ tmp_path, "meta.json" });
    defer allocator.free(meta_path);
    {
        const f = try std.fs.cwd().createFile(meta_path, .{});
        defer f.close();
        try f.writeAll(meta_json);
        try f.sync();
    }

    // hash meta.json
    const hash = try sha256FileHex(allocator, meta_path);
    defer allocator.free(hash);

    // write COMMIT
    const commit_path = try std.fs.path.join(allocator, &.{ tmp_path, "COMMIT" });
    defer allocator.free(commit_path);
    {
        const f = try std.fs.cwd().createFile(commit_path, .{});
        defer f.close();
        try f.writeAll(hash);
        try f.sync();
    }

    // fsync dir (best effort)
    // rename tmp -> final
    try std.fs.cwd().rename(tmp_path, final_path);

    // update latest pointer atomically
    const latest_tmp = try std.fs.path.join(allocator, &.{ base_dir, "latest.tmp" });
    defer allocator.free(latest_tmp);
    const latest_path = try std.fs.path.join(allocator, &.{ base_dir, "latest" });
    defer allocator.free(latest_path);
    {
        const f = try std.fs.cwd().createFile(latest_tmp, .{});
        defer f.close();
        var buf: [32]u8 = undefined;
        const sl = try std.fmt.bufPrint(&buf, "{d}", .{step});
        try f.writeAll(sl);
        try f.sync();
    }
    try std.fs.cwd().rename(latest_tmp, latest_path);
}

// Verify a checkpoint dir has valid COMMIT matching meta.json sha256.
pub fn verifyCommit(allocator: std.mem.Allocator, dir: []const u8) !void {
    const meta_path = try std.fs.path.join(allocator, &.{ dir, "meta.json" });
    defer allocator.free(meta_path);
    const commit_path = try std.fs.path.join(allocator, &.{ dir, "COMMIT" });
    defer allocator.free(commit_path);

    const hash = try sha256FileHex(allocator, meta_path);
    defer allocator.free(hash);

    const commit_file = try std.fs.cwd().openFile(commit_path, .{});
    defer commit_file.close();
    const st = try commit_file.stat();
    const buf = try allocator.alloc(u8, st.size);
    defer allocator.free(buf);
    _ = try commit_file.readAll(buf);

    if (buf.len != hash.len) return error.InvalidCommit;
    if (!std.mem.eql(u8, buf, hash)) return error.InvalidCommit;
}

// Load latest committed step, or null if none.
pub fn loadLatest(allocator: std.mem.Allocator, base_dir: []const u8) !?u64 {
    const latest_path = try std.fs.path.join(allocator, &.{ base_dir, "latest" });
    defer allocator.free(latest_path);
    const file = std.fs.cwd().openFile(latest_path, .{}) catch return null;
    defer file.close();
    const st = try file.stat();
    const buf = try allocator.alloc(u8, st.size);
    defer allocator.free(buf);
    _ = try file.readAll(buf);
    const step = std.fmt.parseInt(u64, buf, 10) catch return null;

    // verify commit
    var name_buf: [128]u8 = undefined;
    const dir_name = try std.fmt.bufPrint(&name_buf, "ckpt-{d}", .{step});
    const dir_path = try std.fs.path.join(allocator, &.{ base_dir, dir_name });
    defer allocator.free(dir_path);
    verifyCommit(allocator, dir_path) catch return null;
    return step;
}

test "atomicSave + loadLatest roundtrip" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/ckpttest";
    std.fs.cwd().deleteTree(base) catch {};
    try std.fs.cwd().makePath(base);

    try atomicSave(allocator, base, 42, "{\"step\":42}");
    const step = try loadLatest(allocator, base);
    try std.testing.expectEqual(@as(?u64, 42), step);

    // verify commit
    var name_buf: [128]u8 = undefined;
    const dir_name = try std.fmt.bufPrint(&name_buf, "ckpt-{d}", .{42});
    const dir_path = try std.fs.path.join(allocator, &.{ base, dir_name });
    defer allocator.free(dir_path);
    try verifyCommit(allocator, dir_path);
}

test "loadLatest rejects missing commit" {
    const allocator = std.testing.allocator;
    const base = "/tmp/scratchpad/ckpttest2";
    std.fs.cwd().deleteTree(base) catch {};
    try std.fs.cwd().makePath(base);
    // create ckpt dir without COMMIT
    try std.fs.cwd().makePath(base ++ "/ckpt-7");
    // write latest
    {
        const f = try std.fs.cwd().createFile(base ++ "/latest", .{});
        defer f.close();
        try f.writeAll("7");
    }
    const step = try loadLatest(allocator, base);
    try std.testing.expectEqual(@as(?u64, null), step);
}

test "writeF32Tensor creates file" {
    const allocator = std.testing.allocator;
    const path = "/tmp/scratchpad/test.safetensors";
    std.fs.cwd().makePath("/tmp/scratchpad") catch {};
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]usize{ 2, 2 };
    try writeF32Tensor(allocator, path, "w", &shape, &data);
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    const st = try f.stat();
    try std.testing.expect(st.size > 0);
}
EOF
echo done