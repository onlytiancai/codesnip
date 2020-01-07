from multiping import MultiPing
mp = MultiPing(["8.8.8.8"])
mp.send()
responses, no_responses = mp.receive(1)
for addr, rtt in responses.items():
        print("%s responded in %f seconds" % (addr, rtt))
