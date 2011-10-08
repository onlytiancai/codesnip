#coding=utf-8
'DNS拼包拆包，大部分代码借鉴自pydns,'

from struct import pack as struct_pack
from struct import unpack as struct_unpack
import random

#查询类型
QTYPE_A = 1
QTYPE_NS = 2
QTYPE_CNAME = 4 
QTYPE_MX = 15 

#class类型
QCLASS_IN = 1

def hexDump(s):
    '以16进制方式打印字节流'
    return ' '.join(['%.2X' % ord(c) for c in s])

class DNSError(Exception): pass

class DNSRequest(object):
    "DNS请求包，只支持1个查询"
    def __init__(self):
        self.buf = ''
        self.id = 0
    def setHeader(self,
            id=None,  #16bit,报文编号
            qr=0,     #1bit,包类型，0代表查询，1代表应答
            opcode=0, #4bit,查询种类，0：标准查询，1：反向查询
            aa=0,     #1bit,是否为权威应答
            tc=0,     #1bit,包体是否截断
            rd=1,     #1bit,查询中指定，1为递归查询
            ra=0,     #1bit,应答中指定，1为递归查询
            z=0,      #3bit,保留字段，必须设置为0
            rcode=0,  #4bit,应答码，0：无差错，1：格式错，2：DNS出错，3：域名不存在
                      #4：DNS不支持这类查询，5：DNS拒绝查询，6-15：保留字段
            qdcount=3,#16bit,查询记录数
            ancount=0,#16bit,回复记录数
            nscount=0,#16bit,权威技术数
            arcount=0 #16bit,格外记录数
            ):
        if not id:
            id = random.randint(0, 65535)
        self.id = id
        self.buf += self._pack16bit(id)
        self.buf += self._pack16bit((qr&1)<<15 | (opcode&0xF)<<11 | (aa&1)<<10
                | (tc&1)<<9 | (rd&1)<<8 | (ra&1)<<7
                | (z&7)<<4 | (rcode&0xF))
        self.buf += self._pack16bit(qdcount)
        self.buf += self._pack16bit(ancount)
        self.buf += self._pack16bit(nscount)
        self.buf += self._pack16bit(arcount)

    def setQuestion(self, qname, qtype=QTYPE_A, qclass=QCLASS_IN):
        for label in qname.split('.'):
            label = str(label)
            l = len(label)
            self.buf += struct_pack('b', l)
            self.buf += struct_pack(str(l)+'s', label)
        self.buf += '\0'
        self.buf += self._pack16bit(qtype) 
        self.buf += self._pack16bit(qclass)
    def req(self):
        return self.buf
    def _pack16bit(self,n):
        return struct_pack('!H', n)

class DNSResponse(object):
    'DNS应答拆包，只解析头'
    def __init__(self, buf):
        self.buf = buf 
        self.offset = 0

        self.id = self.get16bit()
        flags = self.get16bit()
        self.qr, self.opcode, self.aa,self.tc, self.rd, self.ra, \
            self.z, self.rcode = (
                  (flags>>15)&1,
                  (flags>>11)&0xF,
                  (flags>>10)&1,
                  (flags>>9)&1,
                  (flags>>8)&1,
                  (flags>>7)&1,
                  (flags>>4)&7,
                  (flags>>0)&0xF)
        self.qdcount = self.get16bit()
        self.ancount = self.get16bit()
        self.nscount = self.get16bit()
        self.arcount = self.get16bit()
        self.qname = 'www.baidu.com'
    def get16bit(self):
        return struct_unpack('!H',self.getbytes(2))[0]

    def getbytes(self, n):
        s = self.buf[self.offset : self.offset + n]
        if len(s) != n: raise DNSError, 'not enough data left'
        self.offset = self.offset + n
        return s


def test():
    import socket
    req = DNSRequest()
    req.setHeader(id=0xb13b)
    req.setQuestion('www.baidu.com', QTYPE_A, QCLASS_IN)
    req = req.req()
    print u"原始请求:\n%s" % hexDump(req)

    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.sendto(req, ('172.0.0.2', 53))
    (rsp, addr) = client.recvfrom(65535)
    rsp = DNSResponse(rsp)
    print "rcode:%s, qcount:%s, acount:%s" % \
        (rsp.rcode,rsp.qdcount,rsp.ancount)
    client.close()

if __name__ == '__main__':
    test()
