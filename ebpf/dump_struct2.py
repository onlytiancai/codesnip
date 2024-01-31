from elftools.elf.elffile import ELFFile
from pprint import pprint
import sys

elf_file = sys.argv[1] 
struct_name = sys.argv[2].encode('ascii')
def dump_child(target):
    if target.has_children:
        print('struct %s {' % (target.attributes['DW_AT_name'].value.decode('ascii')))
        for ch in target.iter_children():
            if 'DW_AT_type' in ch.attributes:
                ch_name = ch.attributes['DW_AT_name'].value.decode('ascii')
                tag = ch.get_DIE_from_attribute('DW_AT_type').tag
                if tag == 'DW_TAG_base_type':
                    type_name = ch.attributes['DW_AT_name'].value.decode('ascii')
                    print('\t%s %s;' % (type_name, ch_name))
                elif tag == 'DW_TAG_typedef':
                    type_name = ch.get_DIE_from_attribute('DW_AT_type').attributes['DW_AT_name'].value.decode('ascii')
                    print('\t%s %s;' % (type_name, ch_name))
                elif tag == 'DW_TAG_pointer_type':
                    t = ch.get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type')
                    if 'DW_AT_name' in t.attributes:
                        type_name = t.attributes['DW_AT_name'].value.decode('ascii')
                    else:
                        type_name = 'void'
                    print('\t%s *%s;' % (type_name, ch_name))
                elif tag == 'DW_TAG_array_type':
                    type_name = ch.get_DIE_from_attribute('DW_AT_type').get_DIE_from_attribute('DW_AT_type').attributes['DW_AT_name'].value.decode('ascii')
                    upper_bound = list(ch.get_DIE_from_attribute('DW_AT_type').iter_children())[0].attributes['DW_AT_upper_bound'].value
                    print('\t%s %s[%s];' % (type_name, ch_name, upper_bound))
                else:
                    print('\t%s %s' % (
                        type.attributes['DW_AT_name'].value, 
                        'not base type'))
            else:
                print('\t%s %s' % (
                    type.attributes['DW_AT_name'].value, 
                    'can not find type info'))
        print('};')
        sys.exit()

with open(elf_file, 'rb') as file:
    elf_info = ELFFile(file)
    dwarf_info = elf_info.get_dwarf_info()
    for i, cu in enumerate(dwarf_info.iter_CUs()): 
        for die in cu.iter_DIEs():
            if 'DW_AT_name' in die.attributes and die.attributes['DW_AT_name'].value == struct_name:
                dump_child(die)
