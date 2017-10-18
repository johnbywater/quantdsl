"""Unittest main program"""

import sys
import os
import types

# from . import loader, runner
# from .signals import installHandler

# __unittest = True

# FAILFAST     = "  -f, --failfast   Stop on first failure\n"
# CATCHBREAK   = "  -c, --catch      Catch control-C and display results\n"
# BUFFEROUTPUT = "  -b, --buffer     Buffer stdout and stderr during test runs\n"
import datetime
import six

from quantdsl.interfaces.calcandplot import calc_print
from quantdsl.syntax import find_module_path

USAGE_AS_MAIN = """\
Usage: %(progName)s [options] [tests]

Options:
  -h, --help       Show this message
  -v, --verbose    Verbose output
  -q, --quiet      Minimal output

Example:
  %(progName)s mylib.model1

"""


class TestProgram(object):
    """A command-line program that evaluates a module.
    """
    USAGE = USAGE_AS_MAIN

    progName = None

    def __init__(self, module='__main__', argv=None, exit=True, verbosity=1):
        if isinstance(module, six.string_types):
            self.module = __import__(module)
            for part in module.split('.')[1:]:
                self.module = getattr(self.module, part)
        else:
            self.module = module
        if argv is None:
            argv = sys.argv

        self.exit = exit
        self.verbosity = verbosity
        self.progName = os.path.basename(argv[0])
        self.parseArgs(argv)
        self.runTests()

    def usageExit(self, msg=None):
        if msg:
            print(msg)
        usage = {'progName': self.progName}
        print(self.USAGE % usage)
        sys.exit(2)

    def parseArgs(self, argv):
        # if len(argv) > 1 and argv[1].lower() == 'discover':
        #     self._do_discovery(argv[2:])
        #     return

        import getopt
        long_opts = ['help', 'verbose', 'quiet']
        try:
            options, args = getopt.getopt(argv[1:], 'hHvq', long_opts)
            for opt, value in options:
                if opt in ('-h','-H','--help'):
                    self.usageExit()
                if opt in ('-q','--quiet'):
                    self.verbosity = 0
                if opt in ('-v','--verbose'):
                    self.verbosity = 2
            if len(args) == 0:
                self.usageExit('module name required')
            self.testNames = args
            if __name__ == '__main__':
                # to support python -m unittest ...
                self.module = None
            self.createTests()
        except getopt.error as msg:
            self.usageExit(msg)

    def createTests(self):
        pass

    def runTests(self):
        for module_name in self.testNames:
            path = find_module_path(module_name)
            with open(path) as f:
                source_code = f.read()
            calc_print(source_code,
                       observation_date='2011-1-1',
                       verbose=True,
                       )
        print("Evaluating module")
        # if self.testRunner is None:
        #     self.testRunner = runner.TextTestRunner
        # if isinstance(self.testRunner, (type, types.ClassType)):
        #     try:
        #         testRunner = self.testRunner(verbosity=self.verbosity,
        #                                      failfast=self.failfast,
        #                                      buffer=self.buffer)
        #     except TypeError:
        #         # didn't accept the verbosity, buffer or failfast arguments
        #         testRunner = self.testRunner()
        # else:
        #     # it is assumed to be a TestRunner instance
        #     # testRunner = self.testRunner
        # self.result = testRunner.run(self.test)
        # if self.exit:
        #     sys.exit(not self.result.wasSuccessful())

main = TestProgram
