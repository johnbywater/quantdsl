from unittest.case import TestCase

from mock import Mock

from quantdsl.exceptions import DslError


class TestDslError(TestCase):
    def test(self):
        error = DslError(error='error', descr='descr')
        self.assertEqual(error.error, 'error')
        self.assertEqual(error.descr, 'descr')
        self.assertEqual(error.node, None)
        self.assertEqual(error.lineno, None)
        self.assertEqual(repr(error), 'error: descr')

        node = Mock()
        node.lineno = 123
        error = DslError(error='error', descr='descr', node=node)
        self.assertEqual(repr(error), 'error: descr (line 123)')
        self.assertEqual(error.node, node)
        self.assertEqual(error.lineno, 123)
