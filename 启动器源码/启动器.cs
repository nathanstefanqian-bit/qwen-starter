using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

// Qwen Image 2.1 Studio 的便携入口。
// 用 exe 而不是 .lnk：.lnk 里存的是绝对路径，文件夹一换盘/换目录，
// 双击就找不到目标了（Windows 只在同一卷内靠文件 ID 追踪，跨盘无效）；
// exe 每次都从自身所在目录取路径，整个文件夹搬到哪都能用。
class Launcher
{
    [STAThread]
    static void Main()
    {
        string dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
        string bat = Path.Combine(dir, "启动.bat");

        if (!File.Exists(bat))
        {
            MessageBox.Show(
                "启动.bat 不在本程序所在目录：\n" + dir +
                "\n\n请确认整个文件夹是完整解压出来的。",
                "Qwen Image 2.1 Studio",
                MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        var psi = new ProcessStartInfo(bat);
        psi.WorkingDirectory = dir;
        psi.UseShellExecute = true;
        Process.Start(psi);
    }
}
