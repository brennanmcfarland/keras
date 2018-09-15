#import bpy
import c3d

# def do_stuff():
#     bpy.data.objects["Plane"].data.vertices[0].co.x += 2.0
with open('E:\ML\keras\data\humaneva\S1\Mocap_Data\Box_1.c3d', 'rb') as handle:
    reader = c3d.Reader(handle)
    for i, (points, analog) in enumerate(reader.read_frames()):
        print('Frame {}: {}'.format(i, points.round(2)))
