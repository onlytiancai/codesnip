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
using System.Collections.Generic;
using System.Text;

namespace System.IO.Filesystem.Ntfs
{
    public partial class NtfsReader
    {
        /// <summary>
        /// Recurse the node hierarchy and construct its entire name
        /// stopping at the root directory.
        /// </summary>
        private string GetNodeFullNameCore(UInt32 nodeIndex)
        {
            UInt32 node = nodeIndex;

            Stack<UInt32> fullPathNodes = new Stack<UInt32>();
            fullPathNodes.Push(node);

            UInt32 lastNode = node;
            while (true)
            {
                UInt32 parent = _nodes[node].ParentNodeIndex;

                //loop until we reach the root directory
                if (parent == ROOTDIRECTORY)
                    break;

                if (parent == lastNode)
                    throw new InvalidDataException("Detected a loop in the tree structure.");

                fullPathNodes.Push(parent);

                lastNode = node;
                node = parent;
            }

            StringBuilder fullPath = new StringBuilder();
            fullPath.Append(_driveInfo.Name.TrimEnd(new char[] { '\\' }));

            while (fullPathNodes.Count > 0)
            {
                node = fullPathNodes.Pop();

                fullPath.Append(@"\");
                fullPath.Append(GetNameFromIndex(_nodes[node].NameIndex));
            }

            return fullPath.ToString();
        }
    }
}
