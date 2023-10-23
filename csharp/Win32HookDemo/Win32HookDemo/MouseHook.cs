using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

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

        [DllImport("user32.dll", ExactSpelling = true, CharSet = CharSet.Auto)]
        public static extern IntPtr GetParent(IntPtr hWnd);


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
                    RECT rct;

                    var p = NativeMethods.GetCursorPosition();
                    var allChildWindows = new WindowHandleInfo(NativeMethods.GetForegroundWindow()).GetAllChildHandles();
                    foreach (var ptr in allChildWindows)
                    {
                        NativeMethods.GetWindowRect(ptr, out rct);
                        Console.WriteLine("enum window: {0} {1}", rct, NativeMethods.GetControlText(ptr));
                    }

                    IntPtr winHandle = WindowFromPoint(hookEvent.pt);
                    string title = NativeMethods.GetControlText(winHandle);

                    int controlId = GetDlgCtrlID(winHandle);                    
                    StringBuilder className = new StringBuilder(256);
                    NativeMethods.GetClassName(winHandle, className, className.Capacity);
                   
                    NativeMethods.GetWindowRect(winHandle, out rct);
                    Console.WriteLine("ptr={0},cursor={4},class={1}, text={2}, rect={3}", winHandle, className, title, rct, p);

                    IntPtr pptr = winHandle;
                    while ((pptr = GetParent(pptr)) != IntPtr.Zero)
                    {
                        
                        NativeMethods.GetWindowRect(pptr, out rct);
                        IntPtr winHandle2 = WindowFromPoint(hookEvent.pt);
                        string title2 = NativeMethods.GetControlText(winHandle);

                        int controlId2 = GetDlgCtrlID(winHandle);
                        StringBuilder className2 = new StringBuilder(256);
                        NativeMethods.GetClassName(winHandle, className2, className2.Capacity);
                        Console.WriteLine("\tparent ptr={0},class={1}, text={2}, rect={3}", pptr, className2, title2, rct);
                    }


                    //if (className.ToString() == "#32768")
                    //{
                    //    IntPtr hMenu = NativeMethods.GetMenu(winHandle);              

                    //    IntPtr result = GetParent(winHandle);
                    //    if (result != IntPtr.Zero)
                    //    {                      

                    //        hMenu = NativeMethods.GetMenu(result);
                    //        if (hMenu.ToInt32() != 0)
                    //        {
                    //            for (int i = NativeMethods.GetMenuItemCount(hMenu) - 1; i >= 0; i--)
                    //            {
                    //                StringBuilder menuName = new StringBuilder(0x20);
                    //                NativeMethods.GetMenuString(hMenu, (uint)i, menuName, menuName.Capacity, NativeMethods.MF_BYPOSITION);
                    //                Console.WriteLine("menu i={0},text={1}",i, menuName);
                    //            }
                    //        }
                    //    }

                    //}


                    string msg = string.Format("x={0}, y={1}, class={2}, text={3}", hookEvent.pt.X, hookEvent.pt.Y, className, title);
                    if (LeftMouseDown != null) LeftMouseDown(controlId, msg);
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
