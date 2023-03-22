import multiprocessing

def end_func():
    print("p2 finished")
    end = multiprocessing.Value('i',1)
    exit()

def loop_func():
    i=0
    while i<100:
        print("not ended")
        i+=1
    
p1 = multiprocessing.Process(target = loop_func)
p2 = multiprocessing.Process(target = end_func)

p1.start()
p2.start()

p1.join()
p2.join()