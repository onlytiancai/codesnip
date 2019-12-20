using System;
using System.Collections.Generic;
using System.IO.Filesystem.Ntfs;
using System.Windows.Forms;

namespace Everything2
{
    public class ListViewSort : IComparer<ListViewItem>
    {
        private readonly int _col;
        private readonly bool _descK;

        public ListViewSort()
        {
            _col = 0;
        }

        public ListViewSort(int column, object desc)
        {
            _descK = (bool)desc;
            _col = column; //当前列,0,1,2...,参数由ListView控件的ColumnClick事件传递  
        }

        public int Compare(ListViewItem x, ListViewItem y)
        {
            int tempInt = 0;
            INode nodex = (INode)x.Tag;
            INode nodey = (INode)y.Tag;
            switch (this._col)
            {
                case 2:
                    tempInt = Comparer<ulong>.Default.Compare(nodex.Size, nodey.Size);   
                    break;
                case 3:
                    tempInt = Comparer<DateTime>.Default.Compare(nodex.LastChangeTime, nodey.LastChangeTime);
                    break;
                default:
                    tempInt = String.CompareOrdinal(x.SubItems[_col].Text, y.SubItems[_col].Text);
                    break;
            }

            if (_descK)
            {
                return -tempInt;
            }
            return tempInt;
        }
    }
}
