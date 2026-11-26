// Multimodal foundations (PRD 3.13). Minimal vocab layout.

const std = @import("std");

pub const Modality = enum { text, image, audio, video };

pub const VocabLayout = struct {
    text_offset: usize,
    text_size: usize,
    image_offset: usize,
    image_size: usize,
    audio_offset: usize,
    audio_size: usize,
    video_offset: usize,
    video_size: usize,
    special_offset: usize,
    special_size: usize,

    pub fn total(self: *const VocabLayout) usize {
        return self.special_offset + self.special_size;
    }
};

pub const fsq = @import("fsq.zig");
pub const vq = @import("vq.zig");
pub const connector = @import("connector.zig");

pub fn defaultLayout() VocabLayout {
    return .{
        .text_offset = 0,
        .text_size = 49152,
        .image_offset = 49152,
        .image_size = 8192,
        .audio_offset = 57344,
        .audio_size = 8192,
        .video_offset = 65536,
        .video_size = 8192,
        .special_offset = 73728,
        .special_size = 256,
    };
}

test "VocabLayout total" {
    const l = defaultLayout();
    try std.testing.expectEqual(@as(usize, 73984), l.total());
}
