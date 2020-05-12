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
    /// This is where the parts of the file are located on the volume.
    /// </summary>
    public interface IFragment
    {
        /// <summary>
        /// Logical cluster number, location on disk. 
        /// </summary>
        UInt64 Lcn { get; }

        /// <summary>
        /// Virtual cluster number of next fragment.
        /// </summary>
        UInt64 NextVcn { get; }
    }
}
