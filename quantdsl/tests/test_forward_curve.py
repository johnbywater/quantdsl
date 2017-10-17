from unittest.case import TestCase


class SchwartzSmithForwardCurve(dict):
    def __init__(self, config):
        super(SchwartzSmithForwardCurve, self).__init__()
        self.config = config

    def keys(self):
        return self.config.keys()

    def __getitem__(self, item):
        config = self.config[item]
        return []

class TestForwardCurve(TestCase):

    def test(self):
        c = SchwartzSmithForwardCurve({})
        # Check it works like a normal curve dict, with commodity names as keys
        # and values being a list of tuples ('YYYY-MM-DD', price).
        self.assertEqual(list(c.keys()), [])
        self.assertEqual(list(c.values()), [])
        self.assertEqual(list(c.items()), [])
        with self.assertRaises(KeyError):
            c['GAS']

        c = SchwartzSmithForwardCurve({
            'GAS': {},
        })
        self.assertEqual(list(c.keys()), ['GAS'])
        self.assertEqual(c['GAS'], [])
