#!/bin/bash
while true; do
  clear
  echo "=== BusinessCost并行测试监控 ==="
  echo "时间:" $(date)
  echo ""
  echo "📊 进度统计:"
  completed=$(ls outputs/ | grep business_cost_f | wc -l)
  echo "  已完成: $completed/27 ($(($completed * 100 / 27))%)"
  echo "  剩余: $((27 - $completed)) 个组合"
  echo ""
  echo "🔥 进程状态:"
  ps aux | grep test_business_cost | grep -v grep | awk '{print "  PID:" $2 " CPU:" $3 "% MEM:" $6 "MB"}'
  echo ""
  echo "🚀 GPU状态:"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | awk '{print "  GPU利用率:" $1 "% 显存:" $2 "/" $3 "MB 温度:" $4 "°C"}'
  echo ""
  echo "最新完成的组合:"
  ls outputs/ | grep business_cost_f | tail -3
  echo ""
  echo "按Ctrl+C停止监控"
  sleep 30
done 