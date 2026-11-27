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

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var w = file.writer(io, &buf);

    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, padded, .little);
    try w.writeAll(&len_bytes);

    try w.writeAll(json_buf.items);
    var pad: usize = padded - header_len;
    while (pad > 0) : (pad -= 1) try w.writeAll(" ");

    for (data) |v| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, @bitCast(v), .little);
        try w.writeAll(&b);
    }
    try w.flush();
}

// --- Atomic checkpoint protocol (PRD 3.4) ---

pub fn sha256FileHex(allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const st = try file.stat(io);
    const buf = try allocator.alloc(u8, st.size);
    defer allocator.free(buf);
    var rbuf: [4096]u8 = undefined;
    var r = file.reader(io, &rbuf);
    _ = try r.readAll(buf);
    // Use our own SHA-256 instead of std.crypto.
    const crypto = @import("../crypto/sha256.zig");
    const digest = crypto.sha256(buf);
    const hex = try allocator.alloc(u8, 64);
    const hexchars = "0123456789abcdef";
    for (digest, 0..) |byte, i| {
        hex[i * 2] = hexchars[byte >> 4];
        hex[i * 2 + 1] = hexchars[byte & 0x0f];
    }
    return hex;
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

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    // create tmp dir
    try std.Io.Dir.cwd().createDirPath(io, tmp_path);

    // write meta.json
    const meta_path = try std.fs.path.join(allocator, &.{ tmp_path, "meta.json" });
    defer allocator.free(meta_path);
    {
        const f = try std.Io.Dir.cwd().createFile(io, meta_path, .{});
        defer f.close(io);
        var buf: [4096]u8 = undefined;
        var w = f.writer(io, &buf);
        try w.writeAll(meta_json);
        try w.flush();
        try f.sync(io);
    }

    // hash meta.json
    const hash = try sha256FileHex(allocator, meta_path);
    defer allocator.free(hash);

    // write COMMIT
    const commit_path = try std.fs.path.join(allocator, &.{ tmp_path, "COMMIT" });
    defer allocator.free(commit_path);
    {
        const f = try std.Io.Dir.cwd().createFile(io, commit_path, .{});
        defer f.close(io);
        var buf: [4096]u8 = undefined;
        var w = f.writer(io, &buf);
        try w.writeAll(hash);
        try w.flush();
        try f.sync(io);
    }

    // rename tmp -> final
    try std.Io.Dir.cwd().rename(tmp_path, std.Io.Dir.cwd(), final_path, io);

    // update latest pointer atomically
    const latest_tmp = try std.fs.path.join(allocator, &.{ base_dir, "latest.tmp" });
    defer allocator.free(latest_tmp);
    const latest_path = try std.fs.path.join(allocator, &.{ base_dir, "latest" });
    defer allocator.free(latest_path);
    {
        const f = try std.Io.Dir.cwd().createFile(io, latest_tmp, .{});
        defer f.close(io);
        var buf: [32]u8 = undefined;
        const sl = try std.fmt.bufPrint(&buf, "{d}", .{step});
        var wbuf: [4096]u8 = undefined;
        var w = f.writer(io, &wbuf);
        try w.writeAll(sl);
        try w.flush();
        try f.sync(io);
    }
    try std.Io.Dir.cwd().rename(latest_tmp, std.Io.Dir.cwd(), latest_path, io);
}

// Verify a checkpoint dir has valid COMMIT matching meta.json sha256.
pub fn verifyCommit(allocator: std.mem.Allocator, dir: []const u8) !void {
    const meta_path = try std.fs.path.join(allocator, &.{ dir, "meta.json" });
    defer allocator.free(meta_path);
    const commit_path = try std.fs.path.join(allocator, &.{ dir, "COMMIT" });
    defer allocator.free(commit_path);

    const hash = try sha256FileHex(allocator, meta_path);
    defer allocator.free(hash);

    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const commit_file = try std.Io.Dir.cwd().openFile(io, commit_path, .{});
    defer commit_file.close(io);
    const st = try commit_file.stat(io);
    const buf = try allocator.alloc(u8, st.size);
    defer allocator.free(buf);
    var rbuf: [4096]u8 = undefined;
    var r = commit_file.reader(io, &rbuf);
    _ = try r.readAll(buf);

    if (buf.len != hash.len) return error.InvalidCommit;
    if (!std.mem.eql(u8, buf, hash)) return error.InvalidCommit;
}

// Load latest committed step, or null if none.
pub fn loadLatest(allocator: std.mem.Allocator, base_dir: []const u8) !?u64 {
    const latest_path = try std.fs.path.join(allocator, &.{ base_dir, "latest" });
    defer allocator.free(latest_path);
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const file = std.Io.Dir.cwd().openFile(io, latest_path, .{}) catch return null;
    defer file.close(io);
    const st = try file.stat(io);
    const buf = try allocator.alloc(u8, st.size);
    defer allocator.free(buf);
    var rbuf: [4096]u8 = undefined;
    var r = file.reader(io, &rbuf);
    _ = try r.readAll(buf);
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
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    std.Io.Dir.cwd().deleteTree(io, base) catch {};
    try std.Io.Dir.cwd().createDirPath(io, base);

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
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    std.Io.Dir.cwd().deleteTree(io, base) catch {};
    try std.Io.Dir.cwd().createDirPath(io, base);
    // create ckpt dir without COMMIT
    try std.Io.Dir.cwd().createDirPath(io, base ++ "/ckpt-7");
    // write latest
    {
        const f = try std.Io.Dir.cwd().createFile(io, base ++ "/latest", .{});
        defer f.close(io);
        var buf: [4096]u8 = undefined;
        var w = f.writer(io, &buf);
        try w.writeAll("7");
        try w.flush();
    }
    const step = try loadLatest(allocator, base);
    try std.testing.expectEqual(@as(?u64, null), step);
}

test "writeF32Tensor creates file" {
    const allocator = std.testing.allocator;
    const path = "/tmp/scratchpad/test.safetensors";
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();
    std.Io.Dir.cwd().createDirPath(io, "/tmp/scratchpad") catch {};
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]usize{ 2, 2 };
    try writeF32Tensor(allocator, path, "w", &shape, &data);
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const st = try f.stat(io);
    try std.testing.expect(st.size > 0);
}

// --- Watchdog (PRD 3.4) ---

pub const Watchdog = struct {
    start_time: i64,
    interval_sec: i64 = 30 * 60,
    hard_limit_sec: i64 = 11 * 3600 + 50 * 60,
    last_checkpoint: i64,

    pub fn init() Watchdog {
        const now = std.time.timestamp();
        return .{ .start_time = now, .last_checkpoint = now };
    }

    pub fn elapsed(self: *const Watchdog) i64 {
        return std.time.timestamp() - self.start_time;
    }

    pub fn shouldCheckpoint(self: *const Watchdog) bool {
        return (std.time.timestamp() - self.last_checkpoint) >= self.interval_sec;
    }

    pub fn recordCheckpoint(self: *Watchdog) void {
        self.last_checkpoint = std.time.timestamp();
    }

    pub fn shouldStop(self: *const Watchdog) bool {
        return self.elapsed() >= self.hard_limit_sec - 10 * 60; // stop 10 min before hard limit
    }

    pub fn shouldHardStop(self: *const Watchdog) bool {
        return self.elapsed() >= self.hard_limit_sec;
    }
};

// SIGTERM handling: global flag.
var sigterm_received: std.atomic.Value(bool) = std.atomic.Value(bool).init(false);

pub fn sigtermHandler(_: c_int) callconv(.c) void {
    sigterm_received.store(true, .seq_cst);
}

pub fn installSigtermHandler() void {
    const sa = std.posix.Sigaction{
        .handler = .{ .handler = sigtermHandler },
        .mask = std.posix.empty_sigset,
        .flags = 0,
    };
    std.posix.sigaction(std.posix.SIG.TERM, &sa, null) catch {};
}

pub fn sigtermReceived() bool {
    return sigterm_received.load(.seq_cst);
}

test "Watchdog basic" {
    const wd = Watchdog.init();
    try std.testing.expect(wd.elapsed() >= 0);
    try std.testing.expect(!wd.shouldHardStop());
}
