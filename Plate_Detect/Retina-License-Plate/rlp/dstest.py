#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
import os
import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import sys
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import numpy as np
import pyds
import cv2


fps_streams = {}
frame_count = {}
saved_count = {}
PROCESS_CLASS = [0]
TRACK_CLASS_IDS = [0, 2, 3, 5, 7]
COUNTER = 0
LST_PREFIX = []
SGIE = True

TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
# tiler_sink_pad_buffer_probe  will extract metadata received on tiler src pad
# and update params for drawing rectangle, object information etc.
def crop_object(frame, rect_params, min_size = 32):
    
    left, top, width, height = rect_params.left, rect_params.top, rect_params.width, rect_params.height
    if min(width, height) > min_size:
        x1 = int(left)
        y1 = int(top)
        x2 = int(left + width)
        y2 = int(top + height)
        return (x1, y1, x2, y2), frame[y1:y2, x1:x2]
    else:
        return None


def tiler_sink_pad_buffer_probe(pad, info, u_data):
    global COUNTER
    frame_number = 0
    num_rects = 0
    obj_num = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        source_id    = frame_meta.source_id 
        print(frame_number)
        fps_streams["stream{0}".format(source_id)].get_fps()

        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        # convert python array into numpy array format in the copy mode.
        frame_copy = np.array(n_frame, copy=True, order='C')
        # convert the array into cv2 default color format
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        cv2.rectangle(n_frame, (10, 10), (50, 50), (255,0,0), 10)
        # cv2.imwrite("tmp/frame_{}.jpg".format(frame_number), frame_copy)

        # Extract object 
        l_obj = frame_meta.obj_meta_list

        # Extract all object
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_class_id    = obj_meta.class_id
            if obj_class_id in TRACK_CLASS_IDS:
                
                if obj_meta.class_id in PROCESS_CLASS:
                    l_obj_usr = obj_meta.obj_user_meta_list 
                    try:
                        pass
                        # Casting l_obj_usr.data to pyds.NvDsUserMeta
                    except StopIteration:
                        break


                    # try:
                    #     l_obj_usr=l_obj_usr.next
                    # except StopIteration:
                    #     break
            try:
                l_obj=l_obj.next
            except StopIteration:
                break

        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


'''
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    global COUNTER
    frame_number = 0
    num_rects = 0
    obj_num = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        source_id    = frame_meta.source_id 
        fps_streams["stream{0}".format(source_id)].get_fps()

        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
        # convert python array into numpy array format in the copy mode.
        frame_copy = np.array(n_frame, copy=True, order='C')
        # convert the array into cv2 default color format
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGR)
        cv2.rectangle(n_frame, (10, 10), (50, 50), (255,0,0), 10)
        # cv2.imwrite("tmp/frame_{}.jpg".format(frame_number), frame_copy)

        # Extract object 
        l_obj = frame_meta.obj_meta_list

        # Extract all object
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_class_id    = obj_meta.class_id
            if obj_class_id in TRACK_CLASS_IDS:
                
                if obj_meta.class_id in PROCESS_CLASS:
                    l_clsf = obj_meta.classifier_meta_list 
                    while l_clsf is not None:
                        try:
                            # Casting l_obj.data to pyds.NvDsObjectMeta
                            clsf_meta=pyds.NvDsClassifierMeta.cast(l_clsf.data)
                        except StopIteration:
                            break
                        l_label = clsf_meta.label_info_list 
                        
                        while l_label is not None:
                            try:
                                label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                            except StopIteration:
                                break
                        

                            obj_num += 1
                            print(f"{frame_number}: Object={obj_num}")
                            label_ = label_info.result_label
                            label_conf = round(label_info.result_prob, 5)
                            # print(label_info.result_class_id)
                            if label_ == 'no-LP':
                                print("{}: Object={} Class={} Prob={}".format(frame_number, obj_num, label_, label_conf))


                            obj_track_id = obj_meta.object_id

                            
                            if not os.path.exists('cropped/{}'.format(obj_track_id)):
                                os.makedirs('cropped/{}'.format(obj_track_id), exist_ok = True)

                            data = crop_object(frame_copy, obj_meta.rect_params)

                            if data is not None:
                                box, obj_image = data
                                filename  = "cropped/{}/{}.jpg".format(obj_track_id, frame_number)
                                cv2.imwrite(filename, obj_image)
                                x1, y1, x2, y2 = box
                                with open(filename.replace('.jpg', '.txt'), 'w+') as f:
                                    f.write('{} {} {} {}'.format(x1, y1, x2, y2))
                    
                            try: 
                                l_label=l_label.next
                            except StopIteration:
                                break
        
                        try: 
                            l_clsf=l_clsf.next
                        except StopIteration:
                            break
            try:
                l_obj=l_obj.next
            except StopIteration:
                break

        
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

'''


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN] <folder to save frames>\n" % args[0])
        sys.exit(1)

    for i in range(0, len(args) - 1):
        fps_streams["stream{0}".format(i)] = GETFPS(i)
    number_sources = len(args) - 1

    
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        # os.makedirs(folder_name + "/stream_" + str(i), exist_ok = True)
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i + 1]
        LST_PREFIX.append(uri_name.strip('/').split('/')[-1].split('.')[0])
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    print("[INFO] Create tracker")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker")
    config = configparser.ConfigParser()
    config.read('dstest_tracker_config.txt')
    config.sections()

    if SGIE:
        print("Creating Sgie \n ")
        sgie = Gst.ElementFactory.make("nvinfer", "secondary-nvinference-engine")
        if not sgie:
            sys.stderr.write(" Unable to make sgie \n")

    for key in config['tracker']:
        if key == 'tracker-width' :
            tracker_width = config.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        if key == 'enable-past-frame' :
            tracker_enable_past_frame = config.getint('tracker', key)
            tracker.set_property('enable_past_frame', tracker_enable_past_frame)

    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating capsfilter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    capsfilter1 = Gst.ElementFactory.make("capsfilter", "capsfilter1")
    if not capsfilter1:
        sys.stderr.write(" Unable to get the caps capsfilter1 \n")
    capsfilter1.set_property("caps", caps1)
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    # print("Creating nvvidconv \n ")
    # nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    # if not nvvidconv:
    #     sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    nvosd.set_property('process-mode', 1)
    nvosd.set_property('display-text', 1)
    nvosd.set_property('display-clock', 1)
    nvosd.set_property('gpu-id', 0)
    
    if (is_aarch64()):
        print("Creating transform \n ")
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")
    #################################################################

    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    if not nvvidconv2:
        sys.stderr.write(" Unable to create nvvidconv2 \n")

    capsfilter2 = Gst.ElementFactory.make("capsfilter", "capsfilter2")
    if not capsfilter2:
        sys.stderr.write(" Unable to create capsfilter2 \n")

    caps2 = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter2.set_property("caps", caps2)

    print("Creating Encoder \n")
    encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder \n")
    encoder.set_property("bitrate", 2000000)

    print("Creating Code Parser \n")
    codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
    if not codeparser:
        sys.stderr.write(" Unable to create code parser \n")

    print("Creating Container \n")
    container = Gst.ElementFactory.make("qtmux", "qtmux")
    if not container:
        sys.stderr.write(" Unable to create code parser \n")

    print("Creating Sink \n")
    sink = Gst.ElementFactory.make("filesink", "filesink")
    if not sink:
        sys.stderr.write(" Unable to create file sink \n")

    
    # print("Creating fakesink \n")
    # sink = Gst.ElementFactory.make("fakesink", "fakesink")
    # if not sink:
    #     sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1440)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "dstest_pgie_config.txt")
    if SGIE:
        sgie.set_property('config-file-path', "dstest_sgie_config.txt")
    # pgie.set_property('model-engine-file', "weights/object_detection/yolor_csp_x_star-nms.onnx_b4_gpu0_fp16.engine")
    # pgie_batch_size = pgie.get_property("batch-size")
    # if (pgie_batch_size != number_sources):
    #     print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
    #           number_sources, " \n")
    #     pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    sink.set_property("location", "./out.mp4")
    sink.set_property("sync", 0)
    sink.set_property("qos", 0)


    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        nvvidconv2.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tracker)
    if SGIE:
        pipeline.add(sgie)
    pipeline.add(tiler)
    pipeline.add(capsfilter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv2) ############################################################################
    pipeline.add(encoder) ############################################################################
    pipeline.add(capsfilter2) ############################################################################
    pipeline.add(codeparser) ############################################################################
    pipeline.add(container) ############################################################################
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(tracker)
    if SGIE:
        tracker.link(sgie)
        sgie.link(nvvidconv1)
    else:
        tracker.link(nvvidconv1)
    nvvidconv1.link(capsfilter1)
    capsfilter1.link(tiler)
    tiler.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(nvvidconv2)
    else:
        nvosd.link(nvvidconv2)
    nvvidconv2.link(capsfilter2)
    capsfilter2.link(encoder)
    encoder.link(codeparser)

    for i in range(number_sources):
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        padname = "video_%u" % i
        sinkpad2 = container.get_request_pad(padname)
        if not sinkpad2:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad2 = codeparser.get_static_pad("src")
        if not srcpad2:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad2.link(sinkpad2)
    container.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args[:-1]):
        if i != 0:
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events      
    pipeline.set_state(Gst.State.PLAYING)

    loop.run()

    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)
    return True

if __name__ == '__main__':
    sys.exit(main(sys.argv))