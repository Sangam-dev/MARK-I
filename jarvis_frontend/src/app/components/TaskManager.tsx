import { motion } from "motion/react";
import { CheckCircle2, Circle, ListTodo } from "lucide-react";
import { useState } from "react";

const initialTasks = [
  { id: 1, title: "Review code changes", completed: true },
  { id: 2, title: "Update documentation", completed: false },
  { id: 3, title: "Design new feature", completed: false },
  { id: 4, title: "Team sync meeting", completed: true },
  { id: 5, title: "Deploy to staging", completed: false },
];

export function TaskManager() {
  const [tasks, setTasks] = useState(initialTasks);

  const toggleTask = (id: number) => {
    setTasks(tasks.map(task =>
      task.id === id ? { ...task, completed: !task.completed } : task
    ));
  };

  const completedCount = tasks.filter(t => t.completed).length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="rounded-2xl p-6 backdrop-blur-xl"
      style={{
        background: "linear-gradient(135deg, rgba(17, 24, 39, 0.7) 0%, rgba(31, 41, 55, 0.5) 100%)",
        border: "1px solid rgba(255, 255, 255, 0.1)",
      }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <ListTodo className="w-5 h-5 text-emerald-400" />
          <h3 className="text-white/90">Tasks</h3>
        </div>
        <span className="text-xs text-white/50">{completedCount}/{tasks.length}</span>
      </div>

      <div className="mb-4">
        <div className="h-2 bg-white/10 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-emerald-500 to-green-500"
            initial={{ width: 0 }}
            animate={{ width: `${(completedCount / tasks.length) * 100}%` }}
            transition={{ duration: 0.5 }}
          />
        </div>
      </div>

      <div className="space-y-2">
        {tasks.map((task, index) => (
          <motion.button
            key={task.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 + index * 0.05 }}
            onClick={() => toggleTask(task.id)}
            className="w-full flex items-center gap-3 p-3 rounded-lg text-left transition-all"
            style={{
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
            }}
            whileHover={{ x: 4 }}
          >
            {task.completed ? (
              <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
            ) : (
              <Circle className="w-5 h-5 text-white/30 flex-shrink-0" />
            )}
            <span className={`text-sm ${task.completed ? "text-white/50 line-through" : "text-white/90"}`}>
              {task.title}
            </span>
          </motion.button>
        ))}
      </div>
    </motion.div>
  );
}
