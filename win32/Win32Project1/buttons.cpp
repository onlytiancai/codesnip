#include "stdafx.h"

/**
* 按钮风格
* BS_PUSHBUTTON: 指定一个命令按钮
* BS_CHECKBOX: 指定在矩形按钮的右侧带有标题的选择框，不会自动反选
* WS_TABSTOP: tap键可以停留在上面
*/

LRESULT CALLBACK myWndProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
{
	switch (message)
	{
	case WM_CREATE:
	{
		//在这里创建子窗口和控件

		//创建普通按钮
		HWND hwnd_btn = CreateWindow(
			L"Button",
			L"普通按钮",
			WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
			15, 5,
			100, 40,
			hwnd,//指定button按钮的父窗口
			(HMENU)10000, //控件ID号替代menu,控件ID号唯一
			((LPCREATESTRUCT)lparam)->hInstance,
			NULL
		);

		//创建单选按钮
		HWND hwnd_btn2 = CreateWindow(
			L"Button",
			L"单选按钮",
			WS_CHILD | WS_VISIBLE | BS_AUTORADIOBUTTON,
			15, 50,
			100, 40,
			hwnd,//指定button按钮的父窗口
			(HMENU)10001, //控件ID号替代menu,控件ID号唯一
			((LPCREATESTRUCT)lparam)->hInstance,
			NULL
		);

		//复选框按钮
		HWND hwnd_btn3 = CreateWindow(
			L"Button",
			L"复选按钮",
			WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
			15, 95,
			100, 40,
			hwnd,//指定button按钮的父窗口
			(HMENU)10002, //控件ID号替代menu,控件ID号唯一
			((LPCREATESTRUCT)lparam)->hInstance,
			NULL
		);


		//如何识别这些按钮的点击消息?

		//当点击按钮控件的时候，会像父窗口发送命令消息WM_COMMAND

		// BN_CLICKED: 用户在按钮上单击鼠标时会像父窗口发送BN_CLICKED消息.
		// BN_DOUBALECLICKED: 同上，发送双击 BN_DOUBLECLICKED消息.

	}
	break;
	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	case WM_COMMAND:
	{
		//根据控件ID来进行区分
		switch (LOWORD(wparam))//低字节是控件ID
		{
		case 10000:
			if (HIWORD(wparam) == BN_CLICKED) //高位字节是控件触发的消息类型
			{
				MessageBox(hwnd, L"点击了普通按钮!", L"提示", MB_OK);
			}
			break;
		case 10001:
		{
			MessageBox(hwnd, L"点击了单选按钮!", L"提示", MB_OK);

			HWND hbutton_radio = (HWND)lparam;

			//获取单选按钮的选中状态
			if (SendMessage(hbutton_radio, BM_GETCHECK, 0, 0) == BST_CHECKED)
			{
				MessageBox(hwnd, L"单选按钮被选中!", L"提示", MB_OK);
			}
			else if (SendMessage(hbutton_radio, BM_GETCHECK, 0, 0) == BST_UNCHECKED)
			{
				MessageBox(hwnd, L"单选按钮没有被选中!", L"提示", MB_OK);
			}

		}
		break;
		case 10002:
		{
			MessageBox(hwnd, L"点击了复选按钮!", L"提示", MB_OK);

			HWND  hbutton_check = (HWND)lparam;

			//获取复选框的选中状态
			if (SendMessage(hbutton_check, BM_GETCHECK, 0, 0) == BST_CHECKED)
			{
				MessageBox(hwnd, L"复选按钮被选中!", L"提示", MB_OK);
			}
			else if (SendMessage(hbutton_check, BM_GETCHECK, 0, 0) == BST_UNCHECKED)
			{
				MessageBox(hwnd, L"复选按钮没有被选中!", L"提示", MB_OK);
			}

		}
		break;
		default:
			break;
		}
	}
	break;
	default:
		break;
	}

	return DefWindowProc(hwnd, message, wparam, lparam);
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
	//1.注册窗口类
	WNDCLASS wnd;
	wnd.cbClsExtra = 0;
	wnd.cbWndExtra = 0;
	wnd.hbrBackground = (HBRUSH)GetStockObject(GRAY_BRUSH);
	wnd.hCursor = LoadCursor(hInstance, IDC_ARROW);
	wnd.hIcon = LoadIcon(hInstance, IDI_APPLICATION);
	wnd.lpszClassName = L"zhangkai";
	wnd.lpszMenuName = NULL;
	wnd.lpfnWndProc = myWndProc;
	wnd.style = CS_HREDRAW | CS_VREDRAW;
	wnd.hInstance = hInstance;

	int ec = RegisterClass(&wnd);
	if (ec == 0)
	{
		ec = GetLastError();
		return ec;
	}

	//2.创建窗口
	HWND hwnd = CreateWindow(L"zhangkai", L"window title", WS_OVERLAPPEDWINDOW, 0, 0, 500, 500, NULL, NULL, hInstance, NULL);
	if (hwnd == NULL)
	{
		//create window file
		return 0;
	}

	ShowWindow(hwnd, nShowCmd);
	UpdateWindow(hwnd);


	//3.消息循环

	MSG msg;
	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}


	return 0;
}
