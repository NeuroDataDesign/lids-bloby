"""This class is used to upload results to BOSS"""

from intern.remote.boss import BossRemote
from intern.resource.boss.resource import *
import tifffile as tf
import numpy as np
import os
from src.bossmeta import *
import configparser

class IngestTifStack(object):
    """
    IngestTifStack class can be instantiated with the following args

    - **parameters**, **types**, **return** and **return types**::
    :param ingest_args: Object of ConfigParams
    :param verbose: (optional) to print debug messages
    :type ingest_args: ConfigParams
    :type verbose: integer
    """

    def __init__(self, ingest_args, verbose=0):
        self.args = ingest_args
        self.verbose = verbose

    def _upload_to_boss(self, rmt, data, channel_resource, resolution=0):
        Z_LOC = 0
        size = data.shape
        for i in range(0, data.shape[Z_LOC], 16):
            last_z = i+16
            if last_z > data.shape[Z_LOC]:
                last_z = data.shape[Z_LOC]
            if self.verbose > 0: print(resolution, [0, size[2]], [0, size[1]], [i, last_z])
            rmt.create_cutout(channel_resource, resolution, [0, size[2]], [0, size[1]], [i, last_z], np.asarray(data[i:last_z,:,:], order='C'))

    def _get_boss_config(self):
        config = configparser.ConfigParser()
        config.read(self.args.config)
        token = config['Default']['token']
        boss_url = ''.join(
            (config['Default']['protocol'], '://', config['Default']['host']))
        return token, boss_url

    def _get_channel_resource(self, rmt, chan_name, coll_name, exp_name, type='image', base_resolution=0, sources=[], datatype='uint16', new_channel=True):
        channel_resource = ChannelResource(chan_name, coll_name, exp_name, type=type,
                                        base_resolution=base_resolution, sources=sources, datatype=datatype)
        if new_channel:
            new_rsc = rmt.create_project(channel_resource)
            return new_rsc

        return channel_resource

    def start_upload(self, group_name=None):
        """Based on the ingest arguments, uploads the labelled TIF stack to BOSS. *group_name* is an optional parameter to grant access to the newly created channel for a particular group"""
        rmt = BossRemote(cfg_file_or_dict=self.args.config)

        type_to_dtype = {'image': 'uint16', 'annotation': 'uint64'}

        img = tf.imread(os.path.expanduser(self.args.tif_stack))
        if self.args.type == 'annotation' and img.dtype != 'uint64':
            img = np.asarray(img, dtype='uint64')

        coll_name = self.args.collection
        exp_name = self.args.experiment
        chan_name = self.args.channel
        source_chan = []

        if self.args.source_channel != None:
            source_chan = [self.args.source_channel]

        # upload image back to boss
        channel_rsc = self._get_channel_resource(rmt, chan_name, coll_name, exp_name, type=self.args.type, sources=source_chan, datatype=type_to_dtype[self.args.type], new_channel=self.args.new_channel)

        if img.dtype != 'uint64' or img.dtype != 'uint16':
            if self.args.type == 'image':
                img = img.astype('uint16')
            else:
                img = img.astype('uint64')

        self._upload_to_boss(rmt, img, channel_rsc)
        url = 'https://ndwebtools.neurodata.io/ndviz_url/{}/{}/'.format(coll_name, exp_name)

        if group_name:
            self._change_permissions(group_name)

        return url

    def _change_permissions(self, group_name):
        token, boss_url = self._get_boss_config()

        meta = BossMeta(self.args.collection, self.args.experiment, self.args.channel)
        rmt = BossRemoteProxy(boss_url, token, meta)

        if self.args.collection not in rmt.list_collections():
            print('collection {} not found'.format(self.args.collection))
            sys.exit(1)

        if group_name not in rmt.list_groups():
            print('group {} not found'.format(group_name))
            sys.exit(1)

        read_perms = ['read']
        read_vol_perms = ['read_volumetric_data']
        admin_perms = ['add', 'update', 'assign_group', 'remove_group']
        admin_vol_perms = ['add_volumetric_data']
        all_perms = read_perms + admin_perms
        all_vol_perms = read_vol_perms + admin_vol_perms
        rmt.add_permissions(group_name, all_perms, all_vol_perms)

class ConfigParams(object):
    """
    ConfigParams class can be instantiated with a *param_dict* which has the following attirutes

    collection - BOSS collection name

    experiment - BOSS experiment name

    channel - Name of the new channel to be created

    tif_stack - path to the labelled TIF image

    type - annotation/raw_image

    new_channel - whether a new channel has to be created or not

    source_channel - if type is *annotation*, the source channel

    config - path to intern.cfg file

    """

    def __init__(self, param_dict):
        self.collection = param_dict['collection']
        self.experiment = param_dict['experiment']
        self.channel = param_dict['channel']
        self.tif_stack = param_dict['tif_stack']
        self.type = param_dict['type']
        self.new_channel = param_dict['new_channel']
        self.source_channel = param_dict['source_channel']
        self.config = param_dict['config']
