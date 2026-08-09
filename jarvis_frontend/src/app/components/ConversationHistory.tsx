import { motion } from "motion/react";
import { MessageSquare, Clock } from "lucide-react";

const conversations = [
  { id: 1, title: "Morning Briefing", time: "9:00 AM", messages: 12 },
  { id: 2, title: "Project Discussion", time: "11:30 AM", messages: 24 },
  { id: 3, title: "Weather Query", time: "2:34 PM", messages: 4 },
  { id: 4, title: "Task Planning", time: "Yesterday", messages: 18 },
  { id: 5, title: "Code Review Help", time: "Yesterday", messages: 32 },
  { id: 6, title: "Meeting Notes", time: "2 days ago", messages: 8 },
];

export function ConversationHistory() {
  return (
    <div className="h-full rounded-2xl p-6 backdrop-blur-xl" style={{
      background: "linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(31, 41, 55, 0.5) 100%)",
      border: "1px solid rgba(255, 255, 255, 0.1)",
    }}>
      <div className="flex items-center gap-2 mb-6">
        <MessageSquare className="w-5 h-5 text-blue-400" />
        <h3 className="text-white/90">Conversations</h3>
      </div>

      <div className="space-y-2">
        {conversations.map((conv, index) => (
          <motion.button
            key={conv.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            whileHover={{ x: 4 }}
            className="w-full p-3 rounded-lg text-left transition-all"
            style={{
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
            }}
          >
            <div className="flex items-start justify-between mb-1">
              <span className="text-sm text-white/90">{conv.title}</span>
              <span className="text-xs text-white/40">{conv.messages}</span>
            </div>
            <div className="flex items-center gap-1 text-xs text-white/50">
              <Clock className="w-3 h-3" />
              <span>{conv.time}</span>
            </div>
          </motion.button>
        ))}
      </div>
    </div>
  );
}
