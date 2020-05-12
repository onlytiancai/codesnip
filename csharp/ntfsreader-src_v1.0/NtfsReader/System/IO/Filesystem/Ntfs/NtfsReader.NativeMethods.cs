using System.Runtime.InteropServices;
/*
    The NtfsReader library.

    Copyright (C) 2008 Danny Couture

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
  
    For the full text of the license see the "License.txt" file.

    This library is based on the work of Jeroen Kessels, Author of JkDefrag.
    http://www.kessels.com/Jkdefrag/
    
    Special thanks goes to him.
  
    Danny Couture
    Software Architect
*/
using System.Text;
using Microsoft.Win32.SafeHandles;

namespace System.IO.Filesystem.Ntfs
{
    public partial class NtfsReader
    {
        [DllImport("kernel32", CharSet = CharSet.Auto, BestFitMapping = false)]
        private static extern bool GetVolumeNameForVolumeMountPoint(String volumeName, StringBuilder uniqueVolumeName, int uniqueNameBufferCapacity);

        [DllImport("kernel32", CharSet = CharSet.Auto, BestFitMapping = false)]
        private static extern SafeFileHandle CreateFile(string lpFileName, FileAccess fileAccess, FileShare fileShare, IntPtr lpSecurityAttributes, FileMode fileMode, int dwFlagsAndAttributes, IntPtr hTemplateFile);

        [DllImport("kernel32", CharSet = CharSet.Auto)]
        private static extern bool ReadFile(SafeFileHandle hFile, IntPtr lpBuffer, uint nNumberOfBytesToRead, out uint lpNumberOfBytesRead, ref NativeOverlapped lpOverlapped);

        [Serializable]
        private enum FileMode : int
        {
            Append = 6,
            Create = 2,
            CreateNew = 1,
            Open = 3,
            OpenOrCreate = 4,
            Truncate = 5
        }

        [Serializable, Flags]
        private enum FileShare : int
        {
            None = 0,
            Read = 1,
            Write = 2,
            Delete = 4,
            All = Read | Write | Delete
        }

        [Serializable, Flags]
        private enum FileAccess : int
        {
            Read = 1,
            ReadWrite = 3,
            Write = 2
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct NativeOverlapped
        {
            public IntPtr privateLow;
            public IntPtr privateHigh;
            public UInt64 Offset;
            public IntPtr EventHandle;

            public NativeOverlapped(UInt64 offset)
            {
                Offset = offset;
                EventHandle = IntPtr.Zero;
                privateLow = IntPtr.Zero;
                privateHigh = IntPtr.Zero;
            }
        }
    }
}
