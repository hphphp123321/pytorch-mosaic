import skvideo.io
import skvideo.datasets
import os

from assignment1 import Assignment1


def process_video(video_file, output_file, max_frames=None):

    a1 = Assignment1()
    a1.encode_features(False)
    a1.train(False)

    writer = skvideo.io.FFmpegWriter(output_file)
    reader = skvideo.io.FFmpegReader(video_file, inputdict={}, outputdict={})

    for i, frame in enumerate(reader.nextFrame()):
        if max_frames and i > max_frames:
            break

        print('Processing frame {:d}/{:d}'.format(i + 1, reader.inputframenum))
        mosaic = a1.mosaic_fast(frame / 255)
        writer.writeFrame(mosaic * 255)


if __name__ == '__main__':
    os.environ['IMAGEIO_FFMPEG_EXE'] = '/usr/bin/ffmpeg'
    process_video('videos/bunny.mp4', 'output_video.mp4')
