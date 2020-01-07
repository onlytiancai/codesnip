import os
import glob
import shutil
from datetime import datetime
from PIL import Image # python3 -m pip install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple


def file_time(file):
    filename, ext = os.path.splitext(file)
    if ext.lower() in ('.jpg', '.png', '.gif', '.jpeg'):
        # https://stackoverflow.com/questions/23064549/get-date-and-time-when-photo-was-taken-from-exif-data-using-pil
        try:
            info = Image.open(file)._getexif()
            if info:
                # https://www.awaresystems.be/imaging/tiff/tifftags/privateifd/exif/datetimeoriginal.html
                if 36867 in info:
                    DateTimeOriginal = info[36867]
                    if DateTimeOriginal:
                        return DateTimeOriginal.replace(':', '-')[:10]            
        except:
            pass

    ts = os.path.getctime(file)
    return datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')


inpath = r'F:\photo_temp\**\*'
outpath = r'F:\photo'

def move_photo(src_file, time):    
    month = time[:7]
    filename = os.path.basename(src_file)
    dst_dir = os.path.join(outpath, month)
    dst_file = os.path.join(dst_dir, filename)

    print(time, dst_file, src_file)

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    
    shutil.move(src_file, dst_file)

for file in glob.glob(inpath, recursive=True):
    if os.path.isfile(file):
        move_photo(file, file_time(file))