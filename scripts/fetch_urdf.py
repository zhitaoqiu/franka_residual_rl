# filepath: scripts/fetch_urdf.py
import urllib.request
import os

def fetch_panda_urdf():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    target_dir = os.path.join(project_root, "assets", "franka_panda")
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "panda.urdf")

    # 1. 换用 Pinocchio 官方的 example-robot-data 仓库，这个格式最纯净
    # 2. 前面加上 mirror.ghproxy.com 代理，解决国内 raw.githubusercontent 访问失败的问题
    url = "https://mirror.ghproxy.com/https://raw.githubusercontent.com/Gepetto/example-robot-data/master/robots/panda_description/urdf/panda.urdf"
    
    print(f"正在通过镜像站下载标准 panda.urdf...")
    try:
        # 伪装一下 User-Agent，防止被代理站拦截
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            with open(target_path, 'wb') as out_file:
                out_file.write(response.read())
        print(f"✅ 下载成功！已保存至: {target_path}")
    except Exception as e:
        print(f"❌ 下载依然失败: {e}")
        print(f"💡 备用方案: 请直接复制以下链接到浏览器下载，并改名为 panda.urdf 放进 assets/franka_panda/ 目录：\n{url}")

if __name__ == "__main__":
    fetch_panda_urdf()