from elftools.elf.elffile import ELFFile
from pprint import pprint
import sys

elf_file = sys.argv[1] 
struct_name = sys.argv[2].encode('ascii')

def dump_struct(struct_name, dies_name_map, dies_offset_map):
    target = dies_name_map.get(struct_name)
    other_structs = []
    if target and target.has_children:
        print('struct %s {' % (target.attributes['DW_AT_name'].value.decode('ascii')))
        for ch in target.iter_children():
            if 'DW_AT_type' in ch.attributes:
                type = dies_offset_map.get(ch.attributes['DW_AT_type'].value)
                if not type:
                    print('error ch:', ch)
                    sys.exit() 

                if type.tag == 'DW_TAG_base_type':
                    print('    %s %s;' % (
                        type.attributes['DW_AT_name'].value.decode('ascii'), 
                        ch.attributes['DW_AT_name'].value.decode('ascii')))
                elif type.tag == 'DW_TAG_pointer_type':
                    type2 = dies_offset_map.get(type.attributes['DW_AT_type'].value)
                    if type2.tag in ('DW_TAG_base_type'):
                        print('    %s *%s;' % (
                            type2.attributes['DW_AT_name'].value.decode('ascii'), 
                            ch.attributes['DW_AT_name'].value.decode('ascii')))
                    elif type2.tag == 'DW_TAG_structure_type':
                        temp_name = type2.attributes['DW_AT_name'].value
                        print('    struct %s *%s;' % (
                            temp_name.decode('ascii'), 
                            ch.attributes['DW_AT_name'].value.decode('ascii')))
                        other_structs.append(temp_name);
                    else:
                        print('sss', type2.tag)
                else:
                    print('\t%s %s' % (
                        type.attributes['DW_AT_name'].value, 
                        'not base type'))
            else:
                print('\t%s %s' % (
                    type.attributes['DW_AT_name'].value, 
                    'can not find type info'))
        print('};')
        for name in other_structs:
            dump_struct(temp_name, dies_name_map, dies_offset_map)

with open(elf_file, 'rb') as file:
    elf_info = ELFFile(file)
    dwarf_info = elf_info.get_dwarf_info()
    dies_name_map = {}
    dies_offset_map = {}
    for i, cu in enumerate(dwarf_info.iter_CUs()): 
        for die in cu.iter_DIEs():
            dies_offset_map[die.offset] = die
            if 'DW_AT_name' in die.attributes:
                dies_name_map[die.attributes['DW_AT_name'].value] = die
    print(dies_offset_map.keys())
    dump_struct(struct_name, dies_name_map, dies_offset_map)            
