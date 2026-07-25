import { motion } from "motion/react";
import { Mic, MicOff } from "lucide-react";
import { useState } from "react";

export function VoiceOrb() {
  const [isListening, setIsListening] = useState(false);

  return (
    <div className="relative flex items-center justify-center">
      <motion.div
        className="absolute w-64 h-64 rounded-full"
        style={{
          background: "radial-gradient(circle, rgba(139, 92, 246, 0.3) 0%, rgba(59, 130, 246, 0.2) 50%, transparent 70%)",
        }}
        animate={{
          scale: isListening ? [1, 1.2, 1] : 1,
          opacity: isListening ? [0.5, 0.8, 0.5] : 0.3,
        }}
        transition={{
          duration: 2,
          repeat: isListening ? Infinity : 0,
          ease: "easeInOut",
        }}
      />

      <motion.div
        className="absolute w-48 h-48 rounded-full"
        style={{
          background: "radial-gradient(circle, rgba(139, 92, 246, 0.4) 0%, rgba(59, 130, 246, 0.3) 60%, transparent 80%)",
        }}
        animate={{
          scale: isListening ? [1, 1.15, 1] : 1,
          opacity: isListening ? [0.6, 1, 0.6] : 0.4,
        }}
        transition={{
          duration: 1.5,
          repeat: isListening ? Infinity : 0,
          ease: "easeInOut",
          delay: 0.2,
        }}
      />

      <motion.button
        onClick={() => setIsListening(!isListening)}
        className="relative w-32 h-32 rounded-full cursor-pointer flex items-center justify-center"
        style={{
          background: "linear-gradient(135deg, rgba(139, 92, 246, 0.8) 0%, rgba(59, 130, 246, 0.6) 100%)",
          backdropFilter: "blur(20px)",
          boxShadow: isListening
            ? "0 0 60px rgba(139, 92, 246, 0.6), 0 0 100px rgba(59, 130, 246, 0.4)"
            : "0 20px 60px rgba(0, 0, 0, 0.4)",
        }}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        animate={{
          boxShadow: isListening
            ? [
                "0 0 60px rgba(139, 92, 246, 0.6), 0 0 100px rgba(59, 130, 246, 0.4)",
                "0 0 80px rgba(139, 92, 246, 0.8), 0 0 120px rgba(59, 130, 246, 0.6)",
                "0 0 60px rgba(139, 92, 246, 0.6), 0 0 100px rgba(59, 130, 246, 0.4)",
              ]
            : "0 20px 60px rgba(0, 0, 0, 0.4)",
        }}
        transition={{
          duration: 2,
          repeat: isListening ? Infinity : 0,
          ease: "easeInOut",
        }}
      >
        {isListening ? (
          <Mic className="w-12 h-12 text-white" />
        ) : (
          <MicOff className="w-12 h-12 text-white/80" />
        )}
      </motion.button>

      {isListening && (
        <motion.div
          className="absolute -bottom-16 text-center"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
        >
          <p className="text-sm text-white/60">Listening...</p>
        </motion.div>
      )}
    </div>
  );
}
