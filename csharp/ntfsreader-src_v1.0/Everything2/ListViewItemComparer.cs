using System;
using System.Collections.Generic;
using System.Text;
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
}
