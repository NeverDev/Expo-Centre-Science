"""
execute it to make a call graph of the program
"""

from pycallgraph import PyCallGraph, Config
from pycallgraph.output import GraphvizOutput, GephiOutput
from multiprocessing import SimpleQueue, Array, freeze_support
from pycallgraph import GlobbingFilter

from main import MainProgram


if __name__ == '__main__':

    graphviz = GephiOutput()
    config = Config(max_depth=6, groups=True,)
    config.trace_filter = GlobbingFilter(exclude=[
        'pycallgraph.*',
        'ModuleSpec.*',
        '_handle_fromlist',
        '_ModuleLock.*',
        '_find_and_load',
        '_ModuleLockManager.*',
        '_find_and_load_unlocked',
        'cb',
        '_get_module_lock',
        '_find_spec',
        '_load_unlocked',
    ])

    with PyCallGraph(output=graphviz, config=config):
        try:

            freeze_support()
            MainProgram()  # needed for multiprocessing

            print("ended")
        except Exception as e:
            print(str(e))

    print("end")
