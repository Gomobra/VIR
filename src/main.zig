const std = @import("std");
const root = @import("root.zig");

pub fn main() void {
    const args = std.os.argv;
    if (args.len < 2) {
        std.debug.print("vir {s}\nUsage: vir <command> [args]\nCommands: version, train, info\n", .{root.version});
        return;
    }
    const cmd = std.mem.sliceTo(args[1], 0);
    if (std.mem.eql(u8, cmd, "version")) {
        std.debug.print("vir {s}\n", .{root.version});
    } else if (std.mem.eql(u8, cmd, "info")) {
        std.debug.print("vir {s}: modules loaded\n", .{root.version});
    } else if (std.mem.eql(u8, cmd, "train")) {
        const allocator = std.heap.page_allocator;
        var model = root.train.TinyModel.init(allocator, 4, 3, 0.1) catch {
            std.debug.print("init failed\n", .{});
            return;
        };
        defer model.deinit();
        const ids = [_]u32{ 0, 1, 2, 3 };
        const targets = [_]u32{ 1, 2, 3, 0 };
        var step: usize = 0;
        while (step < 5) : (step += 1) {
            const loss = model.trainStep(&ids, &targets) catch {
                std.debug.print("step failed\n", .{});
                return;
            };
            std.debug.print("step {d}: loss={d:.4}\n", .{ step, loss });
        }
        std.debug.print("done\n", .{});
    } else {
        std.debug.print("unknown command: {s}\n", .{cmd});
    }
}
