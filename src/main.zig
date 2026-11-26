const std = @import("std");
const root = @import("root.zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("vir {s}\nUsage: vir <command> [args]\nCommands: version, train, info\n", .{root.version});
        return;
    }

    const cmd = args[1];
    if (std.mem.eql(u8, cmd, "version")) {
        std.debug.print("vir {s}\n", .{root.version});
    } else if (std.mem.eql(u8, cmd, "info")) {
        std.debug.print("vir {s}: {d} modules loaded\n", .{ root.version, 16 });
    } else if (std.mem.eql(u8, cmd, "train")) {
        // Run a tiny training demo
        const train = root.train;
        var model = try train.TinyModel.init(allocator, 4, 3, 0.1);
        defer model.deinit();
        const ids = [_]u32{ 0, 1, 2, 3 };
        const targets = [_]u32{ 1, 2, 3, 0 };
        var step: usize = 0;
        while (step < 5) : (step += 1) {
            const loss = try model.trainStep(&ids, &targets);
            std.debug.print("step {d}: loss={d:.4}\n", .{ step, loss });
        }
        std.debug.print("done\n", .{});
    } else {
        std.debug.print("unknown command: {s}\n", .{cmd});
    }
}
