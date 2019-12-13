using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Filesystem.Ntfs;
using System.Windows.Forms;

namespace Everything2
{
    public class ListViewItemComparer : IComparer<ListViewItem>
    {
        public int Compare(ListViewItem x, ListViewItem y)
        {
            return string.Compare(x.Text, y.Text);                       
        }
    }

    public partial class Form1 : Form
    {
        private List<ListViewItem> allItems;
        private List<ListViewItem> myCache;
        public Form1()
        {
            InitializeComponent();

            allItems = new List<ListViewItem>();
            myCache = allItems;
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            listView1.Columns[1].Width = Convert.ToInt32(0.5 * this.listView1.Width);
            listView1.VirtualMode = true;
            listView1.RetrieveVirtualItem += ListView1_RetrieveVirtualItem;

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

                ListViewItem lvi = new ListViewItem();

                lvi.Text = node.Name;
                lvi.SubItems.Add(node.FullName);
                lvi.SubItems.Add(node.Size.ToString());
                lvi.SubItems.Add(node.LastChangeTime.ToString());

                allItems.Add(lvi);
            }

            allItems.Sort((x, y) => string.Compare(x.Text, y.Text));
            listView1.VirtualListSize = myCache.Count;

            this.toolStripStatusLabel1.Text = string.Format("Directory Count: {0}, File Count {1}", directoryCount, fileCount);
        }

        private void ListView1_RetrieveVirtualItem(object sender, RetrieveVirtualItemEventArgs e)
        {
            if (myCache != null)
            {
                e.Item = myCache[e.ItemIndex];
            }
            else
            {
                //A cache miss, so create a new ListViewItem and pass it back.
                int x = e.ItemIndex * e.ItemIndex;
                e.Item = new ListViewItem(x.ToString());
            }
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            DateTime beforeDT = System.DateTime.Now;

            if (textBox1.Text != string.Empty)
            {
                int index = allItems.BinarySearch(new ListViewItem(textBox1.Text), new ListViewItemComparer());


               
               
                if (index >= 0)
                {
                    int upper;
                    for (upper = index; upper < allItems.Count; upper++)
                    {
                        if (allItems[upper].Text != textBox1.Text) break;
                    }
                    myCache = allItems.GetRange(index, upper - index + 1);
                }
                else
                {
                    myCache = new List<ListViewItem>();
                }
            }
            else {
                myCache = allItems;
            }

            DateTime afterDT = System.DateTime.Now;
            TimeSpan ts = afterDT.Subtract(beforeDT);
            listView1.VirtualListSize = myCache.Count;
            this.toolStripStatusLabel1.Text = string.Format("File Count {0}, Take {1} ms", myCache.Count, ts.TotalMilliseconds);
            listView1.Invalidate();

        }

    }
}
