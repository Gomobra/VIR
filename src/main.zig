const std = @import("std");
const root = @import("root.zig");

pub fn main() !void {
    const stdout = std.fs.File.stdout();
    var buf: [64]u8 = undefined;
    var w = stdout.writer(&buf);
    try w.interface.print("vir {s}\n", .{root.version});
    try w.interface.flush();
}
