from elftools.elf.elffile import ELFFile
from pprint import pprint
import sys

elf_file = sys.argv[1] 
struct_name = sys.argv[2].encode('ascii')

typedefs = []

def get_pointer_type(die):
    ret = ''
    while True:
        if die.tag == 'DW_TAG_pointer_type':
            ret += '*'
        if 'DW_AT_name' in die.attributes:
            ret = die.attributes['DW_AT_name'].value.decode('ascii') +' ' + ret
            break
        else:
            if 'DW_AT_type' in die.attributes:
                die = die.get_DIE_from_attribute('DW_AT_type')
            else:
                ret = 'void ' + ret
                break
    return ret

def dump_pointer(ch, ch_name):
    die = ch.get_DIE_from_attribute('DW_AT_type')
    type_name = get_pointer_type(die)
    return '%s%s;' % (type_name, ch_name)

def dump_typedef(ch, ch_name):
    die = ch.get_DIE_from_attribute('DW_AT_type')
    type_name = die.attributes['DW_AT_name'].value
    tag = die.get_DIE_from_attribute('DW_AT_type').tag
    if tag == 'DW_TAG_typedef':
        typedefs.append(die)

    return '%s %s;' % (type_name.decode('ascii'), ch_name)


def get_die(struct_name):
    for i, cu in enumerate(dwarf_info.iter_CUs()):
        for die in cu.iter_DIEs():
            if 'DW_AT_name' in die.attributes and die.attributes['DW_AT_name'].value == struct_name:
                return die

def dump_base_type(ch, ch_name):
    die = ch.get_DIE_from_attribute('DW_AT_type')
    type_name = die.attributes['DW_AT_name'].value
    if 'DW_AT_bit_offset' in ch.attributes:
        ch_name = '%s:%s' % (ch_name, ch.attributes['DW_AT_bit_offset'].value)
    return '%s %s;' % (type_name.decode('ascii'), ch_name)

def get_array_type(die):
    while True:
        if 'DW_AT_name' in die.attributes:
            return die.attributes['DW_AT_name'].value.decode('ascii')
        if 'DW_AT_type' in die.attributes:
            die = die.get_DIE_from_attribute('DW_AT_type')
        else:
            return '!error'

def dump_array_type(ch, ch_name):
    die = ch.get_DIE_from_attribute('DW_AT_type')
    type_name = get_array_type(die) 
    chs = list(die.iter_children())
    # one-dimensional array 
    upper_bound = chs[0].attributes['DW_AT_upper_bound'].value
    return '%s %s[%s];' % (type_name, ch_name, upper_bound)


def dump_struct_member(die, member_name):
    for ch in die.iter_children():
        ch_name = ch.attributes['DW_AT_name'].value.decode('ascii')
        tag = ch.get_DIE_from_attribute('DW_AT_type').tag
        if ch_name != member_name:
            continue
        if tag == 'DW_TAG_typedef':
            return dump_typedef(ch, ch_name)
        if tag == 'DW_TAG_pointer_type':
            return dump_pointer(ch, ch_name)
        if tag == 'DW_TAG_base_type':
            return dump_base_type(ch, ch_name)
        if tag == 'DW_TAG_array_type':
            return dump_array_type(ch, ch_name)
        print('unknown tag', ch_name, tag)



def print_typedef(die):
    name = die.attributes['DW_AT_name'].value.decode('ascii')
    type_die = die.get_DIE_from_attribute('DW_AT_type')
    if type_die.tag == 'DW_TAG_typedef':
        print_typedef(type_die)
    else:
        pass
        # print(111, type_die.tag)
        # TODO
    type_name  = type_die.attributes['DW_AT_name'].value.decode('ascii')
    print('typedef %s %s;' % (name, type_name))

with open(elf_file, 'rb') as file:
    elf_info = ELFFile(file)
    dwarf_info = elf_info.get_dwarf_info()
    die = get_die(struct_name)
    if not die:
        print('can not find', struct_name)
    print(dump_struct_member(die, 'signature'))
    print(dump_struct_member(die, 'connection'))
    print(dump_struct_member(die, 'ctx'))
    print(dump_struct_member(die, 'read_event_handler'))
    print(dump_struct_member(die, 'lingering_time'))
    print(dump_struct_member(die, 'captures'))
    print(dump_struct_member(die, 'limit_rate'))
    print(dump_struct_member(die, 'count'))
    print(dump_struct_member(die, 'lowcase_header'))

print('*'*10)
for die in typedefs:
    print_typedef(die);
