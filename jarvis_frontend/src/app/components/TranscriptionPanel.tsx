import { motion } from "motion/react";
import { Sparkles } from "lucide-react";

const mockTranscripts = [
  { id: 1, text: "What's the weather like today?", time: "2:34 PM", type: "user" },
  { id: 2, text: "It's currently 72°F and sunny in San Francisco.", time: "2:34 PM", type: "assistant" },
  { id: 3, text: "Set a reminder for tomorrow at 9 AM", time: "2:35 PM", type: "user" },
  { id: 4, text: "Reminder set for tomorrow at 9:00 AM.", time: "2:35 PM", type: "assistant" },
];

export function TranscriptionPanel() {
  return (
    <div className="h-full rounded-2xl p-6 backdrop-blur-xl" style={{
      background: "linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(31, 41, 55, 0.5) 100%)",
      border: "1px solid rgba(255, 255, 255, 0.1)",
    }}>
      <div className="flex items-center gap-2 mb-4">
        <Sparkles className="w-5 h-5 text-violet-400" />
        <h3 className="text-white/90">Real-time Transcription</h3>
      </div>

      <div className="space-y-3 max-h-[400px] overflow-y-auto scrollbar-thin">
        {mockTranscripts.map((transcript, index) => (
          <motion.div
            key={transcript.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className={`p-3 rounded-lg ${
              transcript.type === "user"
                ? "bg-violet-500/20 border border-violet-500/30"
                : "bg-blue-500/20 border border-blue-500/30"
            }`}
          >
            <div className="flex items-start justify-between mb-1">
              <span className="text-xs text-white/50">{transcript.type === "user" ? "You" : "AI"}</span>
              <span className="text-xs text-white/40">{transcript.time}</span>
            </div>
            <p className="text-sm text-white/80">{transcript.text}</p>
          </motion.div>
        ))}
      </div>

      <div className="mt-4 pt-4 border-t border-white/10">
        <div className="flex items-center gap-2">
          <div className="flex-1 h-2 bg-white/10 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-violet-500 to-blue-500"
              animate={{ width: ["0%", "100%", "0%"] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>
          <span className="text-xs text-white/50">Active</span>
        </div>
      </div>
    </div>
  );
}
