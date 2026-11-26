const std = @import("std");
const root = @import("root.zig");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("vir {s}\n", .{root.version});
}
