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
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Filesystem.Ntfs;
using System.Diagnostics;

namespace NtfsReaderSample
{
    class Program
    {
        #region AnalyzeSimilarity

        /// <summary>
        /// Find similar files by grouping those of the exact same size together.
        /// </summary>
        static void AnalyzeSimilarity(IEnumerable<INode> nodes, DriveInfo driveInfo)
        {
            IDictionary<UInt64, List<INode>> sizeAggregate =
                Algorithms.AggregateBySize(nodes, 10 * 1024 * 1024);

            List<UInt64> sizes = new List<UInt64>(sizeAggregate.Keys);
            sizes.Sort();
            sizes.Reverse();    //make the bigger ones appear first

            string targetFile = Path.Combine(driveInfo.Name, "similarfiles.txt");
            using (StreamWriter fs = new StreamWriter(targetFile))
            {
                foreach (UInt64 size in sizes)
                {
                    List<INode> similarNodes = sizeAggregate[size];

                    if (similarNodes.Count <= 1)
                        continue;

                    fs.WriteLine("-----------------------------------------");
                    fs.WriteLine("SIZE: {0}", size);
                    fs.WriteLine("-----------------------------------------");

                    foreach (INode node in similarNodes)
                        fs.WriteLine(
                            string.Format(
                                "Index {0}, {1}, {2}, size {3}, path {4}",
                                node.NodeIndex,
                                (node.Attributes & Attributes.Directory) != 0 ? "Dir" : "File",
                                node.Name,
                                node.Size,
                                node.FullName
                            )
                         );
                }
            }

            Console.WriteLine("File similarities report has been saved to {0}", targetFile);
        }

        #endregion

        #region Analyze Fragmentation

        /// <summary>
        /// Find most fragmented files and group by fragment count.
        /// </summary>
        /// <remarks>
        /// Requires RetrieveMode.Streams and RetrieveMode.Fragments
        /// </remarks>
        static void AnalyzeFragmentation(IEnumerable<INode> nodes, DriveInfo driveInfo)
        {
            //Fragmentation Example
            IDictionary<UInt32, List<INode>> fragmentsAggregate =
                Algorithms.AggregateByFragments(nodes, 2);

            List<UInt32> fragments = new List<UInt32>(fragmentsAggregate.Keys);
            fragments.Sort();
            fragments.Reverse(); //make the most fragmented ones appear first

            string targetFile = Path.Combine(driveInfo.Name, "fragmentation.txt");
            using (StreamWriter fs = new StreamWriter("c:\\fragmentation.txt"))
            {
                foreach (UInt32 fragment in fragments)
                {
                    List<INode> fragmentedNodes = fragmentsAggregate[fragment];

                    fs.WriteLine("-----------------------------------------");
                    fs.WriteLine("FRAGMENTS: {0}", fragment);
                    fs.WriteLine("-----------------------------------------");

                    foreach (INode node in fragmentedNodes)
                        fs.WriteLine(
                            string.Format(
                                "Index {0}, {1}, {2}, size {3}, path {4}, lastModification {5}",
                                node.NodeIndex,
                                (node.Attributes & Attributes.Directory) != 0 ? "Dir" : "File",
                                node.Name,
                                node.Size,
                                node.FullName,
                                node.LastChangeTime.ToLocalTime()
                            )
                         );
                }
            }

            Console.WriteLine("Fragmentation Report has been saved to {0}", targetFile);
        }

        #endregion

        static void Main(string[] args)
        {
            Trace.Listeners.Add(new TextWriterTraceListener(Console.Out));

            DriveInfo driveToAnalyze = new DriveInfo("c");

            NtfsReader ntfsReader = 
                new NtfsReader(driveToAnalyze, RetrieveMode.All);

            IEnumerable<INode> nodes =
                ntfsReader.GetNodes(driveToAnalyze.Name);

            int directoryCount = 0, fileCount = 0;
            foreach (INode node in nodes)
            {
                if ((node.Attributes & Attributes.Directory) != 0)
                    directoryCount++;
                else
                    fileCount++;
            }

            Console.WriteLine(
                string.Format(
                    "Directory Count: {0}, File Count {1}",
                    directoryCount,
                    fileCount
                )
            );

            AnalyzeFragmentation(nodes, driveToAnalyze);

            AnalyzeSimilarity(nodes, driveToAnalyze);

            Console.ReadKey();
        }
    }
}
