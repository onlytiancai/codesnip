from elftools.elf.elffile import ELFFile
from pprint import pprint
import sys

elf_file = sys.argv[1] 
struct_name = sys.argv[2]

typedefs = {} 
structs = {} 

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

def inspect_die(ch, die):
    while True:
        if die.tag == 'DW_TAG_typedef':
            name = die.attributes['DW_AT_name']
            if name not in typedefs:
                typedefs[name] = die
        elif die.tag == 'DW_TAG_structure_type':
            if 'DW_AT_name' in die.attributes:
                name = die.attributes['DW_AT_name']
                if name not in structs:
                    structs[name] = die
            else:
                pass
                # 匿名 struct，在 typedef 里定义
        elif die.tag in ('DW_TAG_base_type', 'DW_TAG_pointer_type', 'DW_TAG_subroutine_type'):
            pass        
        else:
            print('inspect_die:unknown tag', die.tag, die)

        if 'DW_AT_type' in die.attributes:
            die = die.get_DIE_from_attribute('DW_AT_type')
        else:
            break

def dump_typedef(ch, ch_name):
    die = ch.get_DIE_from_attribute('DW_AT_type')
    type_name = die.attributes['DW_AT_name'].value
    inspect_die(ch, die)
    return '%s %s;' % (type_name.decode('ascii'), ch_name)


def get_die(struct_name):
    for i, cu in enumerate(dwarf_info.iter_CUs()):
        for die in cu.iter_DIEs():
            if 'DW_AT_name' in die.attributes and die.attributes['DW_AT_name'].value.decode('ascii') == struct_name:
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


def dump_struct_member(die, ch):
    ch_name = ch.attributes['DW_AT_name'].value.decode('ascii')
    tag = ch.get_DIE_from_attribute('DW_AT_type').tag
    if tag == 'DW_TAG_typedef':
        return dump_typedef(ch, ch_name)
    if tag == 'DW_TAG_pointer_type':
        return dump_pointer(ch, ch_name)
    if tag == 'DW_TAG_base_type':
        return dump_base_type(ch, ch_name)
    if tag == 'DW_TAG_array_type':
        return dump_array_type(ch, ch_name)
    print('unknown tag', ch_name, tag)

alread_parsed = set()
def print_typedef(die):
    name = die.attributes['DW_AT_name'].value.decode('ascii')
    if name in alread_parsed:
        return
    type_die = die.get_DIE_from_attribute('DW_AT_type')
    if type_die.tag == 'DW_TAG_typedef':
        print_typedef(type_die)
    elif type_die.tag == 'DW_TAG_pointer_type':
        if 'DW_AT_type' in type_die.attributes:
            pointer_type_die = type_die.get_DIE_from_attribute('DW_AT_type')
            if pointer_type_die.tag == 'DW_TAG_subroutine_type':
                rettype = 'void'
                params = []
                for ch in pointer_type_die.iter_children():
                    if ch.tag == 'DW_TAG_formal_parameter':
                        ch_type_die = ch.get_DIE_from_attribute('DW_AT_type')
                        if ch_type_die.tag == 'DW_TAG_pointer_type':
                            params.append(get_pointer_type(ch_type_die))
                        if ch_type_die.tag == 'DW_TAG_typedef':
                            name = die.attributes['DW_AT_name'].value.decode('ascii')
                            params.append(name)
                        else:
                            print('unknown ch_type_die tag', ch_type_die.tag)
                    else:
                        print('unknown ch tag', ch.tag)
                print('typedef %s (*%s)(%s);' % (rettype, name, ','.join(params)))
            else:
                print('unknown pointer tag')
    elif type_die.tag == 'DW_TAG_structure_type':
        print('typedef struct {')
        for ch in type_die.iter_children():
            print('   ', dump_struct_member(type_die, ch))
        print('} %s;' % name)
    else:
        if 'DW_AT_name' not in type_die.attributes:
            print(111, name, die, type_die)
            sys.exit()
        type_name  = type_die.attributes['DW_AT_name'].value.decode('ascii')
        print('typedef %s %s;' % (name, type_name))
    alread_parsed.add(name)

with open(elf_file, 'rb') as file:
    elf_info = ELFFile(file)
    dwarf_info = elf_info.get_dwarf_info()
    die = get_die(struct_name)
    if not die:
        print('can not find', struct_name)
    print('struct %s {' % struct_name)
    for ch in die.iter_children():
        print('   ', dump_struct_member(die, ch))
    print('};')

while True:
    if not typedefs:
        break
    typedefs_copy = typedefs.copy()
    typedefs = {}
    for name, die in typedefs_copy.items():
        print_typedef(die);
