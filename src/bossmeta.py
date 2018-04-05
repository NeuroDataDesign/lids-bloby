import requests


BOSS_VERSION = 'v1'


class BossMeta:
    def __init__(self, collection, experiment, channel):
        self._collection = collection
        self._experiment = experiment
        self._channel = channel
        self._session = requests.Session()
        self._session.stream = False

    def session(self):
        return self._session

    def channel(self):
        return self._channel

    def experiment(self):
        return self._experiment

    def collection(self):
        return self._collection


class BossRemoteProxy:
    def __init__(self, boss_url, token, meta):
        self.boss_url = boss_url
        if self.boss_url[-1] != '/':
            self.boss_url += '/'
        self.token = token

        # BossMeta contains col, exp, chn info
        self.meta = meta

    def _get(self, url, headers={}):
        if url[0] == '/':
            url = url[1:]
        headers['Authorization'] = 'Token {}'.format(self.token)
        resp = self.meta.session().get("{}{}".format(
            self.boss_url, url), headers=headers)
        assert resp.status_code == 200
        return resp

    def _post(self, url, headers={}, data={}):
        if url[0] == '/':
            url = url[1:]
        headers['Authorization'] = 'Token {}'.format(self.token)
        resp = self.meta.session().post("{}{}".format(self.boss_url, url),
                                        headers=headers, data=data)
        if resp.status_code != 201:
            print('Failed POST with ', data)
        return resp

    def _patch(self, url, headers={}, data={}):
        if url[0] == '/':
            url = url[1:]
        headers['Authorization'] = 'Token {}'.format(self.token)
        resp = self.meta.session().patch("{}{}".format(self.boss_url, url),
                                         headers=headers, data=data)
        if resp.status_code != 200:
            print('Failed PATCH with ', data)
        return resp

    def _delete(self, url, headers={}, data={}):
        if url[0] == '/':
            url = url[1:]
        headers['Authorization'] = 'Token {}'.format(self.token)
        resp = self.meta.session().delete("{}{}".format(self.boss_url, url),
                                          headers=headers, data=data)
        if resp.status_code != 202:
            print('Failed DEL with ', data)
        return resp

    def query_perms(self, group, collection, experiment=None, channel=None):
        query_url = "{}/permissions/?group={}&collection={}".format(
            BOSS_VERSION, group, collection)

        if experiment is not None:
            query_url = query_url + '&experiment={}'.format(experiment)
            if channel is not None:
                query_url = query_url + '&channel={}'.format(channel)

        r = self._get(query_url)
        resp = r.json()
        # just the perms of the group
        return resp['permission-sets'][0]['permissions']

    def list_data(self, list_url):
        # print(list_url)
        resp = self._get(list_url)
        return resp.json()

    def list_groups(self):
        list_url = "{}/groups/".format(BOSS_VERSION)
        return self.list_data(list_url)['groups']

    def list_collections(self):
        list_url = "{}/collection/".format(BOSS_VERSION)
        return self.list_data(list_url)['collections']

    def list_experiments(self):
        list_url = "{}/collection/{}/experiment/".format(
            BOSS_VERSION, self.meta.collection())
        return self.list_data(list_url)['experiments']

    def list_channels(self, experiment):
        list_url = "{}/collection/{}/experiment/{}/channel".format(
            BOSS_VERSION, self.meta.collection(), experiment)
        return self.list_data(list_url)['channels']

    def add_permissions(self, group, permissions, vol_permissions):
        self._add_del_perms(group, permissions, vol_permissions, 'add')

    def delete_permissions(self, group, permissions, vol_permissions):
        self._add_del_perms(group, permissions, vol_permissions, 'del')

    def _add_del_perms(self, group, permissions, vol_permissions, add_del):
        perm_url = "{}/permissions/".format(BOSS_VERSION)

        # set perm on collection
        data = {'group': group, 'permissions': permissions,
                'collection': self.meta.collection()}
        if add_del == 'add':
            self._post(perm_url, data=data)
        else:
            existing_group_perms = self.query_perms(
                group, self.meta.collection())
            new_perms = diff(existing_group_perms, permissions)
            data['permissions'] = new_perms
            self._patch(perm_url, data=data)

        if self.meta.experiment() is None:
            # set perms for all experiments
            experiments = self.list_experiments()
        else:
            experiments = [self.meta.experiment()]

        for experiment in experiments:
            data = {'group': group, 'permissions': permissions,
                    'collection': self.meta.collection(), 'experiment': experiment}
            if add_del == 'add':
                self._post(perm_url, data=data)
            else:
                existing_group_perms = self.query_perms(
                    group, self.meta.collection(), experiment=experiment)
                new_perms = diff(existing_group_perms, permissions)
                data['permissions'] = new_perms
                self._patch(perm_url, data=data)

            if self.meta.channel() is None:
                # set perms for all channels
                channels = self.list_channels(experiment)
            else:
                channels = [self.meta.channel()]

            for channel in channels:
                data = {'group': group, 'permissions': permissions + vol_permissions,
                        'collection': self.meta.collection(), 'experiment': experiment, 'channel': channel}
                if add_del == 'add':
                    self._post(perm_url, data=data)
                else:
                    existing_group_perms = self.query_perms(
                        group, self.meta.collection(), experiment=experiment, channel=channel)
                    new_perms = diff(existing_group_perms,
                                     permissions + vol_permissions)
                    data['permissions'] = new_perms
                    self._patch(perm_url, data=data)


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]
