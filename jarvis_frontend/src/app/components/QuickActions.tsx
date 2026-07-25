import { motion } from "motion/react";
import { Zap, FileText, Mail, Calendar, Bell, Settings } from "lucide-react";

const actions = [
  { id: 1, icon: FileText, label: "New Note", color: "blue" },
  { id: 2, icon: Mail, label: "Send Email", color: "violet" },
  { id: 3, icon: Calendar, label: "Schedule", color: "emerald" },
  { id: 4, icon: Bell, label: "Reminder", color: "rose" },
];

export function QuickActions() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="rounded-2xl p-6 backdrop-blur-xl"
      style={{
        background: "linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(31, 41, 55, 0.5) 100%)",
        border: "1px solid rgba(255, 255, 255, 0.1)",
      }}
    >
      <div className="flex items-center gap-2 mb-4">
        <Zap className="w-5 h-5 text-yellow-400" />
        <h3 className="text-white/90">Quick Actions</h3>
      </div>

      <div className="grid grid-cols-2 gap-3">
        {actions.map((action, index) => {
          const Icon = action.icon;
          return (
            <motion.button
              key={action.id}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 + index * 0.05 }}
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              className="p-4 rounded-xl text-center transition-all"
              style={{
                background: "rgba(255, 255, 255, 0.05)",
                border: "1px solid rgba(255, 255, 255, 0.1)",
              }}
            >
              <Icon
                className="w-6 h-6 mx-auto mb-2"
                style={{
                  color: action.color === "blue" ? "#60a5fa" :
                         action.color === "violet" ? "#a78bfa" :
                         action.color === "emerald" ? "#34d399" : "#fb7185",
                }}
              />
              <p className="text-xs text-white/80">{action.label}</p>
            </motion.button>
          );
        })}
      </div>
    </motion.div>
  );
}
