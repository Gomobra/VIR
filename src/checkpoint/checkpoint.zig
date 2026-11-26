// Checkpoint (PRD 3.4). Safetensors writer/reader minimal.

const std = @import("std");

pub const SafetensorsHeader = struct {
    // minimal: name -> (dtype, shape, offset)
    pub const Entry = struct {
        name: []const u8,
        dtype: []const u8,
        shape: []usize,
        data_offset: usize,
    };
};

// Write a single f32 tensor to safetensors format (8-byte header len + JSON + data).
pub fn writeF32Tensor(allocator: std.mem.Allocator, path: []const u8, name: []const u8, shape: []const usize, data: []const f32) !void {
    // compute numel
    var numel: usize = 1;
    for (shape) |s| numel *= s;
    std.debug.assert(data.len == numel);

    // build JSON header
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

    // pad header to 8-byte alignment
    const header_len = json_buf.items.len;
    const padded = (header_len + 7) & ~@as(usize, 7);

    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();

    // write 8-byte header length
    var len_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_bytes, padded, .little);
    try file.writeAll(&len_bytes);

    // write JSON + padding
    try file.writeAll(json_buf.items);
    var pad: usize = padded - header_len;
    while (pad > 0) : (pad -= 1) try file.writeAll(" ");

    // write data (little-endian f32)
    for (data) |v| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, @bitCast(v), .little);
        try file.writeAll(&b);
    }
}

test "writeF32Tensor creates file" {
    const allocator = std.testing.allocator;
    const path = "/tmp/scratchpad/test.safetensors";
    // ensure dir
    std.fs.cwd().makePath("/tmp/scratchpad") catch {};
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]usize{ 2, 2 };
    try writeF32Tensor(allocator, path, "w", &shape, &data);
    // verify file exists and size > 0
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    const st = try f.stat();
    try std.testing.expect(st.size > 0);
}
