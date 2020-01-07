using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.IO.Filesystem.Ntfs;
using System.Windows.Forms;

namespace Everything2
{


    public partial class Form1 : Form
    {
        private List<ListViewItem> allItems;
        private List<ListViewItem> myCache;
        private BackgroundWorker bw = new BackgroundWorker();
        private int filecount = 0;
        private ImageList imageListSmallIcon;
        private Dictionary<string, int> iconIndex = new Dictionary<string, int>();

        public Form1()
        {
            InitializeComponent();

            allItems = new List<ListViewItem>();
            myCache = allItems;
            imageListSmallIcon = new ImageList();
            imageListSmallIcon.ImageSize = new Size(16, 16);
        }

        private void Form1_Load(object sender, EventArgs e)
        {            

            listView1.Columns[1].Width = Convert.ToInt32(0.5 * this.listView1.Width);

            bw.WorkerReportsProgress = true;
            bw.DoWork += bw_DoWork;
            bw.RunWorkerCompleted += bw_RunWorkerCompleted;
            bw.ProgressChanged += bw_ProgressChanged;
            toolStripStatusLabel1.Text = "准备扫描中...";
            bw.RunWorkerAsync();

        }

        private void bw_ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            toolStripStatusLabel1.Text = string.Format("正在扫描文件: {0}", e.ProgressPercentage);
        }

        private void bw_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            toolStripStatusLabel1.Text = string.Format("扫描完毕，共{0}文件", filecount);
            listView1.VirtualMode = true;
            listView1.SmallImageList = imageListSmallIcon;
            listView1.RetrieveVirtualItem += ListView1_RetrieveVirtualItem;
            listView1.VirtualListSize = myCache.Count;
        }

        private void bw_DoWork(object sender, DoWorkEventArgs e)
        {
            DriveInfo driveToAnalyze = new DriveInfo("c");
            NtfsReader ntfsReader =
                new NtfsReader(driveToAnalyze, RetrieveMode.All);

            IEnumerable<INode> nodes =
                ntfsReader.GetNodes(driveToAnalyze.Name);

            foreach (INode node in nodes)
            {
                ListViewItem lvi = new ListViewItem();

                lvi.Tag = node;
                lvi.Text = node.Name;
                lvi.SubItems.Add(node.FullName);
                lvi.SubItems.Add(node.Size.ToString());
                lvi.SubItems.Add(node.LastChangeTime.ToString());
             
                allItems.Add(lvi);

                if (filecount++ % 1000 == 0)
                {
                    bw.ReportProgress(filecount);
                }
            }

            allItems.Sort((x, y) => string.Compare(x.Text, y.Text));
            
        }

        private void ListView1_RetrieveVirtualItem(object sender, RetrieveVirtualItemEventArgs e)
        {
            if (myCache != null)
            {
                e.Item = myCache[e.ItemIndex];

                string ext = Path.GetExtension(e.Item.Text);
                if (!iconIndex.ContainsKey(ext)) {
                    Icon smallIcon = GetSystemIcon.GetIconByFileType(ext, false);
                    if (smallIcon == null)
                        smallIcon = GetSystemIcon.GetIconByFileType(Path.GetExtension("baidu.txt"), false);
                    imageListSmallIcon.Images.Add(smallIcon);
                    iconIndex[ext] = imageListSmallIcon.Images.Count - 1;
                }

                e.Item.ImageIndex = iconIndex[ext];
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
            else
            {
                myCache = allItems;
            }

            DateTime afterDT = System.DateTime.Now;
            TimeSpan ts = afterDT.Subtract(beforeDT);
            listView1.VirtualListSize = myCache.Count;
            this.toolStripStatusLabel1.Text = string.Format("File Count {0}, Take {1} ms", myCache.Count, ts.TotalMilliseconds);
            listView1.Invalidate();

        }

        private void listView1_ColumnClick(object sender, ColumnClickEventArgs e)
        {
            if (listView1.Columns[e.Column].Tag == null)
            {
                listView1.Columns[e.Column].Tag = true;
            }
            var tabK = (bool)listView1.Columns[e.Column].Tag;
            listView1.Columns[e.Column].Tag = !tabK;


            myCache.Sort(new ListViewSort(e.Column, listView1.Columns[e.Column].Tag));
            listView1.VirtualListSize = myCache.Count;            
            listView1.Invalidate();
        }

        private void listView1_MouseDoubleClick(object sender, MouseEventArgs e)
        {
            ListViewHitTestInfo info = listView1.HitTest(e.X, e.Y);
            if (info.Item != null)
            {
                showItem(info.Item);
            }
        }

        private static void showItem(ListViewItem item)
        {
            INode node = (INode)item.Tag;
            System.Diagnostics.ProcessStartInfo psi = new System.Diagnostics.ProcessStartInfo("Explorer.exe");
            psi.Arguments = "/e,/select," + node.FullName;
            System.Diagnostics.Process.Start(psi);
        }

        private void listView1_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Enter)
            {
                if (listView1.SelectedIndices != null && listView1.SelectedIndices.Count > 0)
                {
                    ListView.SelectedIndexCollection c = listView1.SelectedIndices;
                    ListViewItem item = listView1.Items[c[0]];
                    showItem(item);
                }
            }
        }
    }
}
