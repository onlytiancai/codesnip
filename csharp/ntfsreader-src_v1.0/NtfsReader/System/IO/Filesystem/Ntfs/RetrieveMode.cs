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
    /// Allow one to retrieve only needed information to reduce memory footprint.
    /// </summary>
    [Flags]
    public enum RetrieveMode
    {
        /// <summary>
        /// Includes the name, size, attributes and hierarchical information only.
        /// </summary>
        Minimal = 0,

        /// <summary>
        /// Retrieve the lastModified, lastAccessed and creationTime.
        /// </summary>
        StandardInformations = 1,

        /// <summary>
        /// Retrieve file's streams information.
        /// </summary>
        Streams = 2,

        /// <summary>
        /// Retrieve file's fragments information.
        /// </summary>
        Fragments = 4,

        /// <summary>
        /// Retrieve all information available.
        /// </summary>
        All = StandardInformations | Streams | Fragments,
    }
}
