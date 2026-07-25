import { motion } from "motion/react";
import { Activity, Cpu, HardDrive, Wifi } from "lucide-react";

const statusCards = [
  { id: 1, icon: Cpu, label: "CPU Usage", value: "42%", color: "blue", status: "optimal" },
  { id: 2, icon: HardDrive, label: "Storage", value: "68%", color: "violet", status: "good" },
  { id: 3, icon: Wifi, label: "Network", value: "125ms", color: "emerald", status: "excellent" },
  { id: 4, icon: Activity, label: "Uptime", value: "99.8%", color: "rose", status: "stable" },
];

export function SystemStatus() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {statusCards.map((card, index) => {
        const Icon = card.icon;
        return (
          <motion.div
            key={card.id}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.1 }}
            className="rounded-2xl p-4 backdrop-blur-xl relative overflow-hidden"
            style={{
              background: "linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(31, 41, 55, 0.5) 100%)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
            }}
          >
            <div className="relative z-10">
              <div className="flex items-center justify-between mb-3">
                <Icon className={`w-5 h-5 text-${card.color}-400`} style={{
                  color: card.color === "blue" ? "#60a5fa" :
                         card.color === "violet" ? "#a78bfa" :
                         card.color === "emerald" ? "#34d399" : "#fb7185",
                }} />
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              </div>
              <p className="text-xs text-white/60 mb-1">{card.label}</p>
              <p className="text-white/90">{card.value}</p>
              <p className="text-xs text-white/40 mt-1 capitalize">{card.status}</p>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}
