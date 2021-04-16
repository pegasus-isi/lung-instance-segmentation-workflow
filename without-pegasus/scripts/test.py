import signal

def SIGTERM_handler(signum, frame):
    print("got SIGTERM")

if __name__=="__main__":
    signal.signal(signal.SIGTERM, SIGTERM_handler)
    while True:
        d = 1