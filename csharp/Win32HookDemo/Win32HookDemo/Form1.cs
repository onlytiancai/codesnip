using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace Win32HookDemo
{
    public partial class Form1 : Form
    {
        private delegate void InvokeCallback(string msg);
        KeyboardHook kh;
        public Form1()
        {
            InitializeComponent();
            kh = new KeyboardHook(true);
            kh.KeyDown += Kh_KeyDown;

            MouseHook.Setup();
            MouseHook.LeftMouseDown += MouseHook_LeftMouseDown;
        }

        private void MouseHook_LeftMouseDown(int controlId, string title)
        {
            if (title.Length > 100) title = title.Substring(0, 100);
            appendText(string.Format("Left Mouse down:{0}", title));
        }

        private void Kh_KeyDown(Keys key, bool Shift, bool Ctrl, bool Alt)
        {

            appendText(string.Format("Key Down:{0}", key.ToString()));
        }

        void appendText(string msg) {
            if (textBox1.InvokeRequired)
            {
                InvokeCallback msgCallback = new InvokeCallback(appendText);
                textBox1.Invoke(msgCallback, new object[] { msg });
            }
            else
            {
                textBox1.Text = DateTime.Now.ToString()+' ' + msg +"\r\n" + textBox1.Text;
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void toolStripSplitButton1_ButtonClick(object sender, EventArgs e)
        {
            textBox1.Text = "";
        }
    }
}
