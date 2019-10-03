"""
Framework for small realtime blocks

Copyright (c) July 2019, Brandon Carlson
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of the FreeBSD Project.
"""
import warnings
try:
    from rtlsdr import RtlSdr
except ImportError:
    warnings.warn("Please install the helpers extras for full functionality", ImportWarning)

import asyncio

# TODO -- this design is tentative!
# TODO -- does any inheritance make any sense among sources (DataGen), Sinks (PyAudio TBD), and the ThreadedSequence?
# TODO -- not sure how much of this is "pythonic"


class ThreadedSequence(object):
    generic_instance = 1
    """
    Sequentially execute a linear sequence of blocks on a single thread
    A data frame gets pushed through a series of blocks sequentially
    """
    def __init__(self, name=""):
        self.run = False
        self.sequence = []
        self.name = name
        self.input_queue = asyncio.Queue()
        self.output_queues = []
        if self.name == "":
            self.name = "SEQUENCE "+str(ThreadedSequence.generic_instance)
            ThreadedSequence.generic_instance += 1

    def define_sequence(self, sequence_of_blocks):
        """
        define the list of blocks to run in order.
        each block must contain process() call
        """
        # TODO -- maybe hold callbacks, not entire object?
        self.sequence = sequence_of_blocks

    def add_output_queues(self, queues):
        """
        add output queues to the pre-existing queue list
        """
        for queue in queues:
            self.output_queues.append(queue)

    async def process_async(self):
        """
        Just keep processing data until told to stop
        """
        print(self.name + " started running")
        self.run = True
        while self.run:
            data = await self.input_queue.get()
            if data is None:
                self.run = False
                break
            data = self.process(data)
            for queue in self.output_queues:
                await queue.put(data)
        print(self.name + " stopped running")

    def process(self, data):
        for block in self.sequence:
            data = block.process(data)
        return data

    def stop(self):
        """
        kill processing
        """
        self.run = False
        print(self.name + " commanded to stop running")
        # wake self up
        try:
            self.input_queue.put_nowait(None)
        except asyncio.QueueFull:
            # if the queue is full, it should already be awake!
            pass


class DataGenerator(object):
    """
    This is a data test generator to help test without an RTLSDR
    The timing stability has not been tested, so it may not work with transmit yet!
    """
    def __init__(self, bit_rate=100):
        self.run = False
        self.bit_rate = bit_rate
        self.next_bit = 0
        self.output_queues = []

    def add_output_queues(self, queues):
        """
        add output queues to the pre-existing queue list
        """
        for queue in queues:
            self.output_queues.append(queue)

    async def process_async(self):
        """
        generate data -- TODO just alternating 1/0 now... include PN later
        """
        print("Data Gen Started")
        self.run = True
        while self.run:
            await asyncio.sleep(1)
            data = []
            for i in range(0, self.bit_rate):
                data.append(self.next_bit)
                self.next_bit = 0 if self.next_bit == 1 else 1
            for queue in self.output_queues:
                await queue.put(data)
        print("Data Gen Stopped")

    def stop(self):
        """
        kill processing
        """
        self.run = False
        # wake self up
