import apa102
import time
import threading

try:
    import queue as Queue
except ImportError:
    import Queue as Queue

class Event_Light:
    PIXELS_N = 3
    COLOR_RED = [255, 0, 0]
    COLOR_GREEN = [0, 255, 0]
    COLOR_NONE = [0, 0, 0]


    def __init__(self):
        self.dev = apa102.APA102(num_led=self.PIXELS_N)
        
        self.next = threading.Event()
        self.queue = Queue.Queue()
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
        self.event_Off()


    def event_On(self):
        self.next.set()
        self.queue.put(self._event_On)


    def event_Off(self):
        self.next.set()
        self.queue.put(self._event_Off)


    def light_Off(self):
        self.next.set()
        self.queue.put(self._light_Off)


    def _run(self):
        while True:
            func = self.queue.get()
            func()


    def _event_On(self):
        colors = [self.COLOR_RED, self.COLOR_NONE, self.COLOR_NONE]
        self.next.clear()
        self.write(colors)


    def _event_Off(self):
        colors = [self.COLOR_NONE, self.COLOR_NONE, self.COLOR_GREEN]
        self.next.clear()
        self.write(colors)


    def _light_Off(self):
        colors = [self.COLOR_NONE, self.COLOR_NONE, self.COLOR_NONE]
        self.next.clear()
        self.write(colors)


    def write(self, colors):
        for i in range(self.PIXELS_N):
            self.dev.set_pixel(i, int(colors[i][0]), int(colors[i][1]), int(colors[i][2]), bright_percent=10)
        self.dev.show()


if __name__ == '__main__':
    light = Event_Light()
    time.sleep(3)
    light.event_On()
    time.sleep(3)
    light.light_Off()
    time.sleep(1)
