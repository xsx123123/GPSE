import sys
import os
import subprocess

# 将当前项目的路径加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_target():
    # 这个函数将在子进程中运行
    from gpse.convert.processor import GenomicDataProcessor
    processor = GenomicDataProcessor(plink_path="false")
    result = processor.process_genomic_data(vcf="dummy.vcf", out_prefix="dummy_out/prefix")
    print(f"WORKFLOW_RESULT:{result}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--run-target":
        run_target()
        return

    # 在主进程中，以子进程方式运行自己，并捕获 stderr
    cmd = [sys.executable, __file__, "--run-target"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    stderr_output = res.stderr
    stdout_output = res.stdout
    
    print("--- Subprocess Stdout ---")
    print(stdout_output)
    print("--- Subprocess Stderr ---")
    print(stderr_output)
    print("-------------------------")
    
    # 校验 stderr_output 是否包含错误提示
    has_error = "status 1" in stderr_output or "Error" in stderr_output or "CalledProcessError" in stderr_output
    assert has_error, "Should log the Command error message"
    
    # 校验是否输出了 Traceback 堆栈信息
    has_traceback = "Traceback (most recent call last)" in stderr_output
    if has_traceback:
        print("FAIL: Traceback found in log output!")
        sys.exit(1)
    else:
        print("SUCCESS: Traceback is successfully suppressed!")
        sys.exit(0)

if __name__ == "__main__":
    main()


