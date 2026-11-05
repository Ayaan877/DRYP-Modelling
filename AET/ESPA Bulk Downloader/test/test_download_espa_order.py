import os
import re
import unittest
from mock import patch
import download_espa_order as deo
import urllib
import requests
import requests_mock


class TestDownloadEspaOrder(unittest.TestCase):

    def test_get_version(self):
        ver = deo.get_version()
        self.assertTrue(re.match("[0-9]{1}.[0-9]{1,2}.[0-9]{1,2}", ver))

class TestRequestsHandler(unittest.TestCase):

    def setUp(self):
        rh = deo.RequestsHandler(host='http://foo.gov')
        self.handler = rh

    def tearDown(self):
        self.handler = None
        
        if os.path.exists("bar.tar.part"):
            os.remove("bar.tar.part")

    def test_auth(self):
        self.handler.auth('foo', 'bar')
        self.assertEqual(('foo', 'bar'), self.handler.creds)

    def test_get(self):
        with requests_mock.mock() as m:
            m.get('http://foo.gov/bar', json={'bilbo': 'baggins'})
            resp = self.handler.get('/bar')
            self.assertEqual(resp, {'bilbo': 'baggins'})

    def test_download(self):
        with requests_mock.mock() as m:
            m.head('http://foo.gov/bar.tar', headers={'Content-Length': '99'})
            m.get('http://foo.gov/bar.tar', text="guacamole")
            resp = self.handler.download('/bar.tar', 'bar.tar')
            self.assertEqual(resp, "bar.tar")

class TestApi(unittest.TestCase):

    def setUp(self):
        self.api = deo.Api('bilbo', 'baggins', 'http://foo.gov')

    def tearDown(self):
        self.api = None

    def test_api_request(self):
        with requests_mock.mock() as m:
            m.get('http://foo.gov/morefoo', json={'thing1': 'thing2', 'messages': {'value': 'true'}})
            resp = self.api.api_request('/morefoo')
            self.assertEqual(resp, {'thing1': 'thing2'})

    def test_get_completed_scenes(self):
        with requests_mock.mock() as m:
            m.get('http://foo.gov/api/v1/item-status/anespaorderid', json={'anespaorderid': [{'product_dload_url': 'a.gov'}, 
                                                                                             {'product_dload_url': 'b.gov'}]})
            resp = self.api.get_completed_scenes('anespaorderid')
            self.assertEqual(resp, ['a.gov', 'b.gov'])

    def test_retrieve_all_orders(self):
        with requests_mock.mock() as m:
            m.get('http://foo.gov/api/v1/list-orders/ausersemail@gmail.com', json={'orderjson': {'status': 'complete'}})
            resp = self.api.retrieve_all_orders('ausersemail@gmail.com')
            self.assertEqual(resp, {'orderjson': {'status': 'complete'}})

class TestScene(unittest.TestCase):

    def setUp(self):
        self.test_url = "https://downloads.usgs.gov/orders/user@contractor.gov-030/LC080070592016082701T1-SC20200423155912.tar.gz"

    def test_init(self):
        scene = deo.Scene(self.test_url)
        self.assertEqual(scene.srcurl, self.test_url)
        self.assertEqual(scene.orderid, 'user@contractor.gov-030')
        self.assertEqual(scene.filename, 'LC080070592016082701T1-SC20200423155912.tar.gz')
        self.assertEqual(scene.name, 'LC080070592016082701T1-SC20200423155912')
        self.assertEqual(scene.cksum_url, 'https://downloads.usgs.gov/orders/user@contractor.gov-030/LC080070592016082701T1-SC20200423155912.md5')
        self.assertEqual(scene.cksum_file, 'LC080070592016082701T1-SC20200423155912.md5')
        self.assertEqual(scene.cksum_name, 'LC080070592016082701T1-SC20200423155912 MD5 checksum')

    
class TestLocalStorage(unittest.TestCase):

    def setUp(self):
        self.test_url = "https://downloads.usgs.gov/orders/user@contractor.gov-030/LC080070592016082701T1-SC20200423155912.tar.gz"
        self.scene = deo.Scene(self.test_url)
        self.local_storage = deo.LocalStorage('')

    def tearDown(self):
        path = "user@contractor.gov-030"
        if os.path.exists(path):
            os.rmdir(path)

    def test_directory_path(self):
        path = self.local_storage.directory_path(self.scene)
        self.assertTrue(os.path.exists(path))

    def test_scene_path(self):
        path = self.local_storage.scene_path(self.scene)
        self.assertEqual(path, "user@contractor.gov-030/LC080070592016082701T1-SC20200423155912.tar.gz")

    @patch('os.path.exists', lambda i: type(i) == str)
    def test_is_stored(self):
        self.assertTrue(self.local_storage.is_stored(self.scene))
 
