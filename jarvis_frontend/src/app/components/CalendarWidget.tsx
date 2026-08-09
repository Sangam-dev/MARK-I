import { motion } from "motion/react";
import { Calendar, Clock } from "lucide-react";

const events = [
  { id: 1, title: "Team Standup", time: "9:00 AM", color: "violet" },
  { id: 2, title: "Design Review", time: "11:30 AM", color: "blue" },
  { id: 3, title: "Lunch Break", time: "12:30 PM", color: "emerald" },
  { id: 4, title: "Client Meeting", time: "3:00 PM", color: "rose" },
];

export function CalendarWidget() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="rounded-2xl p-6 backdrop-blur-xl"
      style={{
        background: "linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(31, 41, 55, 0.5) 100%)",
        border: "1px solid rgba(255, 255, 255, 0.1)",
      }}
    >
      <div className="flex items-center gap-2 mb-4">
        <Calendar className="w-5 h-5 text-violet-400" />
        <h3 className="text-white/90">Today's Schedule</h3>
      </div>

      <div className="space-y-2">
        {events.map((event, index) => (
          <motion.div
            key={event.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 + index * 0.05 }}
            className="flex items-center gap-3 p-3 rounded-lg"
            style={{
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
            }}
          >
            <div
              className={`w-1 h-12 rounded-full bg-${event.color}-500`}
              style={{
                background: event.color === "violet" ? "#8b5cf6" :
                           event.color === "blue" ? "#3b82f6" :
                           event.color === "emerald" ? "#10b981" : "#f43f5e",
              }}
            />
            <div className="flex-1">
              <p className="text-sm text-white/90">{event.title}</p>
              <div className="flex items-center gap-1 text-xs text-white/50 mt-1">
                <Clock className="w-3 h-3" />
                <span>{event.time}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
