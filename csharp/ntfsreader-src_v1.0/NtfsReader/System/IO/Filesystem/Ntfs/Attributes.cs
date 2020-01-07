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
namespace System.IO.Filesystem.Ntfs
{
    /// <summary>
    /// Node attributes.
    /// </summary>
    [Flags]
    public enum Attributes : uint
    {
        /// <summary>
        /// The file is read-only.
        /// </summary>
        ReadOnly = 1,

        /// <summary>
        /// The file is hidden, and thus is not included in an ordinary directory listing.
        /// </summary>
        Hidden = 2,

        /// <summary>
        /// The file is a system file. The file is part of the operating system or is
        /// used exclusively by the operating system.
        /// </summary>
        System = 4,

        /// <summary>
        /// The file is a directory.
        /// </summary>
        Directory = 16,

        /// <summary>
        /// The file's archive status. Applications use this attribute to mark files
        /// for backup or removal.
        /// </summary>
        Archive = 32,

        /// <summary>
        /// Reserved for future use.
        /// </summary>
        Device = 64,

        /// <summary>
        /// The file is normal and has no other attributes set. This attribute is valid
        /// only if used alone.
        /// </summary>
        Normal = 128,

        /// <summary>
        /// The file is temporary. File systems attempt to keep all of the data in memory
        /// for quicker access rather than flushing the data back to mass storage. A
        /// temporary file should be deleted by the application as soon as it is no longer
        /// needed.
        /// </summary>
        Temporary = 256,

        /// <summary>
        /// The file is a sparse file. Sparse files are typically large files whose data
        /// are mostly zeros.
        /// </summary>
        SparseFile = 512,

        /// <summary>
        /// The file contains a reparse point, which is a block of user-defined data
        /// associated with a file or a directory.
        /// </summary>
        ReparsePoint = 1024,

        /// <summary>
        /// The file is compressed.
        /// </summary>
        Compressed = 2048,

        /// <summary>
        /// The file is offline. The data of the file is not immediately available.
        /// </summary>
        Offline = 4096,

        /// <summary>
        /// The file will not be indexed by the operating system's content indexing service.
        /// </summary>
        NotContentIndexed = 8192,

        /// <summary>
        /// The file or directory is encrypted. For a file, this means that all data
        /// in the file is encrypted. For a directory, this means that encryption is
        /// the default for newly created files and directories.
        /// </summary>
        Encrypted = 16384,
    }
}
