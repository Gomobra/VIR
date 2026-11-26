const std = @import("std");
const root = @import("root.zig");

pub fn main() void {
    std.debug.print("vir {s}\n", .{root.version});
    const allocator = std.heap.page_allocator;
    var model = root.train.TinyModel.init(allocator, 4, 3, 0.1) catch {
        std.debug.print("init failed\n", .{});
        return;
    };
    defer model.deinit();
    const ids = [_]u32{ 0, 1, 2, 3 };
    const targets = [_]u32{ 1, 2, 3, 0 };
    var step: usize = 0;
    while (step < 3) : (step += 1) {
        const loss = model.trainStep(&ids, &targets) catch {
            std.debug.print("step failed\n", .{});
            return;
        };
        std.debug.print("step {d}: loss={d:.4}\n", .{ step, loss });
    }
    std.debug.print("done\n", .{});
}
