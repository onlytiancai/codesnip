using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Win32HookDemo
{
    public class MouseHook
    {
        public delegate void LeftMouseDownEventHandler(int controlId, string title);
        public static event LeftMouseDownEventHandler LeftMouseDown;

        public static HookProc MouseHookProcedure;
        public static int hHook = 0;

        [DllImport("user32.dll")]
        static extern IntPtr WindowFromPoint(POINT Point);

        [DllImport("user32.dll")]
        static extern int GetDlgCtrlID(IntPtr winHandle);

        
        public static int MouseHookProc(int nCode, IntPtr wParam, IntPtr lParam)
        {
            MouseHookStruct hookEvent = (MouseHookStruct)
                Marshal.PtrToStructure(lParam, typeof(MouseHookStruct));

            if (nCode >= 0)
            {
                // Get the mouse WM from the wParam parameter
                var wmMouse = (MouseMessage)wParam;
                if (wmMouse == MouseMessage.WM_LBUTTONDOWN )
                {
                    IntPtr winHandle = WindowFromPoint(hookEvent.pt);
                    string title = NativeMethods.GetControlText(winHandle);
                    int controlId = GetDlgCtrlID(winHandle);
                    if (LeftMouseDown != null) LeftMouseDown(controlId, title);
                }
            } 
            return NativeMethods.CallNextHookEx(hHook, nCode, wParam, lParam);
        }

        public static void Setup() {
            if (MouseHook.hHook == 0)
            {

                MouseHookProcedure = new HookProc(MouseHookProc);
                hHook = NativeMethods.SetWindowsHookEx(HookType.WH_MOUSE_LL,
                    MouseHookProcedure,
                    Marshal.GetHINSTANCE(Assembly.GetExecutingAssembly().GetModules()[0]),
                    0);
                if (hHook == 0)
                {
                    Console.WriteLine(Marshal.GetLastWin32Error());
                }
                Console.WriteLine("SetWindowsHookEx:{0}", hHook);
            }
        }

        public static void DeactivateMouseHook()
        {
            bool ret = NativeMethods.UnhookWindowsHookEx(hHook);
        }
    }
}
