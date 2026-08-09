import { motion } from "motion/react";
import { Cloud, Droplets, Wind } from "lucide-react";

export function WeatherWidget() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-2xl p-6 backdrop-blur-xl relative overflow-hidden"
      style={{
        background: "linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(139, 92, 246, 0.2) 100%)",
        border: "1px solid rgba(255, 255, 255, 0.1)",
      }}
    >
      <div className="absolute top-0 right-0 w-32 h-32 bg-blue-400/20 rounded-full blur-3xl" />

      <div className="relative z-10">
        <div className="flex items-start justify-between mb-4">
          <div>
            <p className="text-xs text-white/60 mb-1">San Francisco, CA</p>
            <h2 className="text-5xl mb-1 text-white">72°</h2>
            <p className="text-sm text-white/70">Partly Cloudy</p>
          </div>
          <Cloud className="w-12 h-12 text-white/60" />
        </div>

        <div className="grid grid-cols-2 gap-4 mt-6">
          <div className="flex items-center gap-2">
            <Droplets className="w-4 h-4 text-blue-300" />
            <div>
              <p className="text-xs text-white/50">Humidity</p>
              <p className="text-sm text-white/90">65%</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Wind className="w-4 h-4 text-blue-300" />
            <div>
              <p className="text-xs text-white/50">Wind</p>
              <p className="text-sm text-white/90">12 mph</p>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
